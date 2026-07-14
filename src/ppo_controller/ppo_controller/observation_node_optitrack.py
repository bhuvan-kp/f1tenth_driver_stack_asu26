"""OptiTrack-based observation node (no lidar).

Publishes state = [s, ey, epsi, delta, v, yaw_rate, beta] + lookahead
curvatures at 1m/2m/4m ahead, computed from OptiTrack pose (position +
orientation only -- no twist), with:
  - an SE(2) origin offset to align the OptiTrack world frame with the
    map/track frame
  - finite-difference + low-pass-filtered velocity/yaw-rate estimation
    (OptiTrack gives no twist)

SIM MODE: delta (steering) is taken from the last COMMANDED steering angle
on /drive, since there's no separate feedback sensor in sim. Swap the
subscription below back to true hardware steering feedback
(std_msgs/Float32 on /steering_feedback) before deploying on the real car.
"""

import pathlib

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from ppo_interfaces.msg import PPOObservation
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .track_utils import Track, lookahead_curvatures


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class SE2Offset:
    def __init__(self, dx: float, dy: float, dtheta: float):
        self.dx = dx
        self.dy = dy
        self.dtheta = dtheta
        self._cos = np.cos(dtheta)
        self._sin = np.sin(dtheta)

    def apply(self, x: float, y: float, yaw: float):
        x_new = self._cos * x - self._sin * y + self.dx
        y_new = self._sin * x + self._cos * y + self.dy
        yaw_new = wrap_angle(yaw + self.dtheta)
        return x_new, y_new, yaw_new


class PoseVelocityEstimator:
    def __init__(self, cutoff_hz: float = 10.0, expected_rate_hz: float = 120.0):
        dt_nominal = 1.0 / expected_rate_hz
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        self._alpha = dt_nominal / (rc + dt_nominal)

        self._have_prev = False
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._prev_yaw = 0.0
        self._prev_t = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

    def update(self, x: float, y: float, yaw: float, t: float):
        if not self._have_prev:
            self._prev_x, self._prev_y, self._prev_yaw, self._prev_t = x, y, yaw, t
            self._have_prev = True
            return

        dt = t - self._prev_t
        if dt <= 1e-6:
            return

        dx_world = x - self._prev_x
        dy_world = y - self._prev_y
        dyaw = wrap_angle(yaw - self._prev_yaw)

        vx_world = dx_world / dt
        vy_world = dy_world / dt
        yaw_rate_raw = dyaw / dt

        c, s = np.cos(yaw), np.sin(yaw)
        vx_body_raw = c * vx_world + s * vy_world
        vy_body_raw = -s * vx_world + c * vy_world

        a = self._alpha
        self.vx = a * vx_body_raw + (1 - a) * self.vx
        self.vy = a * vy_body_raw + (1 - a) * self.vy
        self.yaw_rate = a * yaw_rate_raw + (1 - a) * self.yaw_rate

        self._prev_x, self._prev_y, self._prev_yaw, self._prev_t = x, y, yaw, t


class ObservationNode(Node):

    def __init__(self):
        super().__init__("observation_node_optitrack")

        self.declare_parameter("origin_offset_x", 0.0)
        self.declare_parameter("origin_offset_y", 0.0)
        self.declare_parameter("origin_offset_theta", 0.0)
        self.offset = SE2Offset(
            self.get_parameter("origin_offset_x").value,
            self.get_parameter("origin_offset_y").value,
            self.get_parameter("origin_offset_theta").value,
        )
        self.get_logger().info(
            f"Origin offset: dx={self.offset.dx}, dy={self.offset.dy}, "
            f"dtheta={self.offset.dtheta} (update via params once calibrated)"
        )

        self.declare_parameter("optitrack_rate_hz", 120.0)
        self.declare_parameter("velocity_cutoff_hz", 10.0)
        self.vel_estimator = PoseVelocityEstimator(
            cutoff_hz=self.get_parameter("velocity_cutoff_hz").value,
            expected_rate_hz=self.get_parameter("optitrack_rate_hz").value,
        )

        # --- SIM MODE: commanded steering stand-in for true feedback ---
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("steering_command_timeout_sec", 0.5)
        self.last_steering = 0.0
        self.last_steering_stamp = None
        drive_topic = self.get_parameter("drive_topic").value
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped,
            drive_topic,
            self.drive_callback,
            10,
        )
        self.get_logger().info(
            f"SIM MODE: using commanded steering from {drive_topic} as delta "
            f"(swap to true hardware feedback before deploying)"
        )

        pkg_share = get_package_share_directory("ppo_controller")
        centerline_csv = pathlib.Path(pkg_share) / "data" / "RRMini_centerline.csv"
        self.track = Track.from_centerline_file(str(centerline_csv))

        self.obs_pub = self.create_publisher(PPOObservation, "/ppo_observation", 10)
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/f1tenth/pose",
            self.pose_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "ObservationNode ready (OptiTrack pose only, no lidar, "
            "lookahead curvatures enabled, SE2 offset applied, SIM steering mode)"
        )

    def drive_callback(self, msg: AckermannDriveStamped):
        self.last_steering = float(msg.drive.steering_angle)
        self.last_steering_stamp = self.get_clock().now()

    def pose_callback(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw_raw = np.arctan2(siny_cosp, cosy_cosp)

        x_raw = msg.pose.position.x
        y_raw = msg.pose.position.y

        x, y, yaw = self.offset.apply(x_raw, y_raw, yaw_raw)

        self.vel_estimator.update(x, y, yaw, t)
        vx, vy = self.vel_estimator.vx, self.vel_estimator.vy
        yaw_rate = self.vel_estimator.yaw_rate
        v = np.hypot(vx, vy)
        beta = np.arctan2(vy, vx) if v > 1e-3 else 0.0

        s, ey, epsi = self.track.cartesian_to_frenet(x, y, yaw)

        delta = self.last_steering
        timeout = self.get_parameter("steering_command_timeout_sec").value
        if self.last_steering_stamp is not None:
            age = (self.get_clock().now() - self.last_steering_stamp).nanoseconds * 1e-9
            if age > timeout:
                self.get_logger().warn(
                    f"Commanded steering stale ({age:.2f}s old, timeout={timeout}s)",
                    throttle_duration_sec=1.0,
                )
        else:
            self.get_logger().warn(
                "No commanded steering received yet, using delta=0.0",
                throttle_duration_sec=1.0,
            )

        state = np.array([s, ey, epsi, delta, v, yaw_rate, beta], dtype=np.float32)
        lookahead = lookahead_curvatures(self.track, s).astype(np.float32)  # 1m, 2m, 4m ahead

        obs = np.concatenate([state, lookahead])

        out = PPOObservation()
        out.state = obs.tolist()
        self.obs_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObservationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()