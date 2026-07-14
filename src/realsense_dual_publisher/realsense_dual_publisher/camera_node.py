#!/usr/bin/env python3
"""
camera_node.py

Opens a RealSense pipeline for the color sensor and the two infrared (IR)
sensors on the D435i and publishes each stream to its own ROS 2 topic:

    /camera/color/image_raw   (sensor_msgs/Image, bgr8)
    /camera/infra1/image_raw  (sensor_msgs/Image, mono8)
    /camera/infra2/image_raw  (sensor_msgs/Image, mono8)

Note: when both IR streams are enabled, the D435i's IR projector is
disabled automatically by librealsense unless emitter_enabled is forced,
since the projector's dot pattern would otherwise show up in both images.
This node disables the emitter explicitly for clean stereo IR.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import pyrealsense2 as rs
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraNode(Node):
    def __init__(self):
        super().__init__('realsense_camera_node')

        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('enable_emitter', False)

        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        enable_emitter = self.get_parameter('enable_emitter').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', qos)
        self.infra1_pub = self.create_publisher(Image, '/camera/infra1/image_raw', qos)
        self.infra2_pub = self.create_publisher(Image, '/camera/infra2/image_raw', qos)

        self.bridge = CvBridge()

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if enable_emitter else 0.0)

        self.get_logger().info(
            f'RealSense color/infra1/infra2 streaming at {width}x{height}@{fps}fps'
        )

        # Poll at roughly 2x the sensor frame rate so we don't silently
        # miss frames while still yielding control back to the executor.
        poll_period = 1.0 / (2.0 * fps)
        self.timer = self.create_timer(poll_period, self._poll_and_publish)

    def _poll_and_publish(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return

        stamp = self.get_clock().now().to_msg()

        color_frame = frames.get_color_frame()
        if color_frame:
            img = np.asanyarray(color_frame.get_data())
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera_color_optical_frame'
            self.color_pub.publish(msg)

        infra1_frame = frames.get_infrared_frame(1)
        if infra1_frame:
            img = np.asanyarray(infra1_frame.get_data())
            msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera_infra1_optical_frame'
            self.infra1_pub.publish(msg)

        infra2_frame = frames.get_infrared_frame(2)
        if infra2_frame:
            img = np.asanyarray(infra2_frame.get_data())
            msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera_infra2_optical_frame'
            self.infra2_pub.publish(msg)

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
