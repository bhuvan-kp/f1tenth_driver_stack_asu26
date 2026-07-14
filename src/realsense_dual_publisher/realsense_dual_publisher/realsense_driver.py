#!/usr/bin/env python3
"""
realsense_driver.py

IMPORTANT: this replaces running camera_node.py and imu_node.py as two
separate OS processes. On Jetson, librealsense is built with
FORCE_RSUSB_BACKEND=true (required to avoid patching the L4T kernel), and
that backend claims the entire USB device exclusively per-process. Two
independent processes each opening their own rs.pipeline() on the same
physical D435i will race: whichever starts first gets the device, and the
second gets "RuntimeError: No device connected".

The fix: a single process opens ONE rs.pipeline() with every stream
enabled (color, infra1, infra2, accel, gyro) and dispatches incoming
frames to two separate rclpy.Node instances, each of which only knows
about its own topics:

    CameraPublisherNode ('realsense_camera_node')
        /camera/color/image_raw
        /camera/infra1/image_raw
        /camera/infra2/image_raw

    ImuPublisherNode ('realsense_imu_node')
        /camera/imu

Both nodes still show up separately in `ros2 node list`, and the three
data types still go to separate topics -- the only change is that they
share one physical device handle within one executable, which is what
the RSUSB backend actually allows.
"""

import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import pyrealsense2 as rs
import numpy as np

from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge


class CameraPublisherNode(Node):
    """Only owns publishers + message construction. Does not touch the
    RealSense pipeline directly -- frames are pushed in from the driver."""

    def __init__(self):
        super().__init__('realsense_camera_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', qos)
        self.infra1_pub = self.create_publisher(Image, '/camera/infra1/image_raw', qos)
        self.infra2_pub = self.create_publisher(Image, '/camera/infra2/image_raw', qos)
        self.bridge = CvBridge()

    def publish_color(self, frame):
        img = np.asanyarray(frame.get_data())
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        self.color_pub.publish(msg)

    def publish_infra(self, index, frame):
        img = np.asanyarray(frame.get_data())
        msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f'camera_infra{index}_optical_frame'
        (self.infra1_pub if index == 1 else self.infra2_pub).publish(msg)


class ImuPublisherNode(Node):
    """Only owns the /camera/imu publisher + message construction."""

    def __init__(self):
        super().__init__('realsense_imu_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.imu_pub = self.create_publisher(Imu, '/camera/imu', qos)

        self._last_gyro = (0.0, 0.0, 0.0)
        self._last_accel = (0.0, 0.0, 0.0)
        self._have_gyro = False
        self._have_accel = False

    def handle_motion(self, stream_type, data):
        if stream_type == rs.stream.gyro:
            self._last_gyro = (data.x, data.y, data.z)
            self._have_gyro = True
        elif stream_type == rs.stream.accel:
            self._last_accel = (data.x, data.y, data.z)
            self._have_accel = True

        if self._have_gyro and self._have_accel:
            self._publish()

    def _publish(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_imu_optical_frame'

        msg.angular_velocity.x = self._last_gyro[0]
        msg.angular_velocity.y = self._last_gyro[1]
        msg.angular_velocity.z = self._last_gyro[2]

        msg.linear_acceleration.x = self._last_accel[0]
        msg.linear_acceleration.y = self._last_accel[1]
        msg.linear_acceleration.z = self._last_accel[2]

        # D435i does not provide fused orientation; mark as unknown.
        msg.orientation_covariance[0] = -1.0

        self.imu_pub.publish(msg)


class RealsenseDriver:
    """Owns the single rs.pipeline() and fans frames out to both nodes."""

    def __init__(self, camera_node, imu_node, width=640, height=480, fps=30,
                 accel_rate=200, gyro_rate=400, enable_emitter=False):
        self.camera_node = camera_node
        self.imu_node = imu_node
        self._lock = threading.Lock()

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_rate)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_rate)

        self._seen_streams = set()

        profile = self.pipeline.start(config, self._frame_callback)

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if enable_emitter else 0.0)

    def _frame_callback(self, frame):
        # Runs on a librealsense-internal thread, not the rclpy executor
        # thread. Publisher.publish() is thread-safe for this use case, but
        # we still serialize with a lock since video and motion callbacks
        # can arrive concurrently on different internal threads.
        #
        # IMPORTANT: color and stereo (infra1/infra2) frames come from two
        # physically separate sensors with independent timestamp domains.
        # librealsense's software syncer does NOT always bundle them into a
        # single frameset -- color in particular frequently arrives as a
        # standalone video frame. We must handle both delivery shapes or
        # frames get silently dropped (no error, nothing published).
        with self._lock:
            if frame.is_frameset():
                for f in frame.as_frameset():
                    self._dispatch_single_frame(f)
            else:
                self._dispatch_single_frame(frame)

    def _dispatch_single_frame(self, f):
        if f.is_video_frame():
            profile = f.get_profile()
            stream_type = profile.stream_type()
            if stream_type == rs.stream.color:
                self._log_first_seen('color')
                self.camera_node.publish_color(f)
            elif stream_type == rs.stream.infrared:
                idx = profile.stream_index()
                self._log_first_seen(f'infra{idx}')
                self.camera_node.publish_infra(idx, f)
        elif f.is_motion_frame():
            motion = f.as_motion_frame()
            data = motion.get_motion_data()
            stream_type = f.get_profile().stream_type()
            self._log_first_seen('gyro' if stream_type == rs.stream.gyro else 'accel')
            self.imu_node.handle_motion(stream_type, data)

    def _log_first_seen(self, name):
        if name not in self._seen_streams:
            self._seen_streams.add(name)
            self.camera_node.get_logger().info(f'First {name} frame received from device')

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)

    camera_node = CameraPublisherNode()
    imu_node = ImuPublisherNode()

    # A plain node just to hold parameters, so they can be set from the
    # launch file / CLI without touching this script.
    config_node = Node('realsense_driver_config')
    config_node.declare_parameter('width', 640)
    config_node.declare_parameter('height', 480)
    config_node.declare_parameter('fps', 30)
    config_node.declare_parameter('accel_rate', 200)
    config_node.declare_parameter('gyro_rate', 400)
    config_node.declare_parameter('enable_emitter', False)

    width = config_node.get_parameter('width').value
    height = config_node.get_parameter('height').value
    fps = config_node.get_parameter('fps').value
    accel_rate = config_node.get_parameter('accel_rate').value
    gyro_rate = config_node.get_parameter('gyro_rate').value
    enable_emitter = config_node.get_parameter('enable_emitter').value

    driver = RealsenseDriver(
        camera_node, imu_node,
        width=width, height=height, fps=fps,
        accel_rate=accel_rate, gyro_rate=gyro_rate,
        enable_emitter=enable_emitter,
    )
    camera_node.get_logger().info(
        f'RealSense color/infra1/infra2 streaming at {width}x{height}@{fps}fps'
    )
    imu_node.get_logger().info(
        f'RealSense IMU streaming (accel {accel_rate}Hz, gyro {gyro_rate}Hz)'
    )

    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    executor.add_node(imu_node)
    executor.add_node(config_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        camera_node.destroy_node()
        imu_node.destroy_node()
        config_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
