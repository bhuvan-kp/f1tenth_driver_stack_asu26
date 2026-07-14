import pathlib
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from ppo_interfaces.msg import PPOObservation
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .track_utils import Track, lookahead_curvatures
from .scan_simulator import MapDistanceField, simulate_scan


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class SE2Offset:
    """Rigid transform from OptiTrack world frame -> map/track frame.

    x_map = R(theta) @ (x_opti - origin) is NOT what we want; the offset
    describes where the OptiTrack origin sits *in* the map frame, so:

        p_map = R(theta) @ p_opti + [dx, dy]
        yaw_map = yaw_opti + theta
    """

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
    """Finite-difference velocity/yaw-rate from pose-only input, EMA filtered.

    OptiTrack gives pose at ~120 Hz with no twist. Raw finite differences at
    that rate are noisy enough to inject jitter straight into the policy, so
    we low-pass filter with a single-pole EMA. alpha is derived from a
    cutoff frequency rather than hardcoded, so it stays correct if the
    publish rate ever changes.
    """

    def __init__(self, cutoff_hz: float = 10.0, expected_rate_hz: float = 120.0):
        dt_nominal = 1.0 / expected_rate_hz
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        self._alpha = dt_nominal / (rc + dt_nominal)  # EMA smoothing factor

        self._have_prev = False
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._prev_yaw = 0.0
        self._prev_t = 0.0

        self.vx = 0.0     # body-frame longitudinal velocity
        self.vy = 0.0     # body-frame lateral velocity
        self.yaw_rate = 0.0

    def update(self, x: float, y: float, yaw: float, t: float):
        if not self._have_prev:
            self._prev_x, self._prev_y, self._prev_yaw, self._prev_t = x, y, yaw, t
            self._have_prev = True
            return

        dt = t - self._prev_t
        if dt <= 1e-6:
            # Duplicate/out-of-order timestamp; skip this update rather than
            # dividing by ~0 and blowing up the filter.
            return

        dx_world = x - self._prev_x
        dy_world = y - self._prev_y
        dyaw = wrap_angle(yaw - self._prev_yaw)

        vx_world = dx_world / dt
        vy_world = dy_world / dt
        yaw_rate_raw = dyaw / dt

        # Rotate world-frame velocity into body frame using the *current* yaw.
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
        super().__init__("observation_node")

        # --- SE(2) origin offset (OptiTrack world -> map/track frame) ---
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

        # --- velocity estimation (pose-only input) ---
        self.declare_parameter("optitrack_rate_hz", 120.0)
        self.declare_parameter("velocity_cutoff_hz", 10.0)
        self.vel_estimator = PoseVelocityEstimator(
            cutoff_hz=self.get_parameter("velocity_cutoff_hz").value,
            expected_rate_hz=self.get_parameter("optitrack_rate_hz").value,
        )

        # --- scan params (must match training) ---
        self.declare_parameter("num_beams", 64)
        self.declare_parameter("fov", 4.7)
        self.declare_parameter("max_range", 10.0)
        self.declare_parameter("scan_eps", 0.01)
        self.num_beams = self.get_parameter("num_beams").value
        self.fov = self.get_parameter("fov").value
        self.max_range = self.get_parameter("max_range").value
        self.scan_eps = self.get_parameter("scan_eps").value

        # --- map + track ---
        pkg_share = get_package_share_directory("ppo_controller")
        map_yaml = pathlib.Path(pkg_share) / "data" / "rand_test.yaml"
        centerline_csv = pathlib.Path(pkg_share) / "data" / "rand_test_centerline.csv"

        self.distance_field = MapDistanceField(str(map_yaml))
        self.track = Track.from_centerline_file(str(centerline_csv))

        # --- pub/sub ---
        self.obs_pub = self.create_publisher(PPOObservation, "/ppo_observation", 10)
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/optitrack/pose",
            self.pose_callback,
            qos_profile_sensor_data,
        )

        self.prev_steer = 0.0  # if delta (steering) is part of state and not
                                 # directly measurable, track last commanded value
        self.get_logger().info("ObservationNode ready (pose-only, SE2 offset applied)")

    def pose_callback(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Extract yaw from quaternion (planar assumption, yaw only)
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw_raw = np.arctan2(siny_cosp, cosy_cosp)

        x_raw = msg.pose.position.x
        y_raw = msg.pose.position.y

        # Apply origin offset BEFORE anything else touches the pose, so
        # velocity estimation and the frenet/scan lookups are all done in
        # map frame consistently.
        x, y, yaw = self.offset.apply(x_raw, y_raw, yaw_raw)

        self.vel_estimator.update(x, y, yaw, t)
        vx, vy = self.vel_estimator.vx, self.vel_estimator.vy
        yaw_rate = self.vel_estimator.yaw_rate
        v = np.hypot(vx, vy)
        beta = np.arctan2(vy, vx) if v > 1e-3 else 0.0

        # Frenet conversion
        s, ey, epsi = self.track.cartesian_to_frenet(x, y, yaw)

        # delta (steering angle) isn't observable from pose alone; using
        # last commanded steering as a proxy is the standard approach here.
        # If your obs vector needs a *measured* steering angle instead,
        # this is the one field that still needs a sensor (steering pot,
        # servo feedback, etc).
        delta = self.prev_steer

        state = np.array([s, ey, epsi, delta, v, yaw_rate, beta], dtype=np.float32)

        scan = simulate_scan(
            x, y, yaw,
            self.distance_field,
            num_beams=self.num_beams,
            fov=self.fov,
            max_range=self.max_range,
            eps=self.scan_eps,
        ).astype(np.float32)

        lookahead = lookahead_curvatures(self.track, s).astype(np.float32)  # 1m, 2m, 4m ahead

        obs = np.concatenate([state, scan, lookahead])

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
