#!/usr/bin/env python3
"""
imu_node.py

Opens a second, independent RealSense pipeline that only enables the
motion module (accelerometer + gyroscope) on the D435i and publishes
combined IMU data to:

    /camera/imu   (sensor_msgs/Imu)

The D435i exposes its vision sensors (color/IR) and its motion module as
separate USB interfaces on the same physical device, so this pipeline can
run concurrently with camera_node.py's pipeline without conflict.

Accel and gyro frames arrive asynchronously and at different native rates,
so this node keeps the latest sample of each and publishes a combined Imu
message every time either one updates.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import pyrealsense2 as rs

from sensor_msgs.msg import Imu


class ImuNode(Node):
    def __init__(self):
        super().__init__('realsense_imu_node')

        self.declare_parameter('accel_rate', 200)
        self.declare_parameter('gyro_rate', 400)

        accel_rate = self.get_parameter('accel_rate').value
        gyro_rate = self.get_parameter('gyro_rate').value

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

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_rate)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_rate)
        self.pipeline.start(config, self._frame_callback)

        self.get_logger().info(
            f'RealSense IMU streaming (accel {accel_rate}Hz, gyro {gyro_rate}Hz)'
        )

    def _frame_callback(self, frame):
        # This callback runs on a librealsense internal thread, not the
        # rclpy executor thread. Publishing from here is fine since
        # rclpy publishers are thread-safe for this simple use case.
        if frame.is_motion_frame():
            motion = frame.as_motion_frame()
            data = motion.get_motion_data()
            stream_type = frame.get_profile().stream_type()

            if stream_type == rs.stream.gyro:
                self._last_gyro = (data.x, data.y, data.z)
                self._have_gyro = True
            elif stream_type == rs.stream.accel:
                self._last_accel = (data.x, data.y, data.z)
                self._have_accel = True

            if self._have_gyro and self._have_accel:
                self._publish_imu(frame.get_timestamp())

    def _publish_imu(self, rs_timestamp_ms):
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

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
