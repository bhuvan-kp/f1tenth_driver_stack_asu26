#!/usr/bin/env python3
"""
Pure pursuit controller for F1TENTH.

Subscribes to OptiTrack pose (geometry_msgs/PoseStamped) instead of
odometry. Since OptiTrack provides no twist, forward velocity and yaw
rate are estimated by finite-differencing consecutive poses (using
message timestamps) and low-pass filtering the result.

An SE(2) origin offset (translation + rotation) can be applied to align
the OptiTrack world frame with the map/track frame the waypoints are
defined in -- same convention as observation_node_optitrack.py.

Loads a centerline waypoint file (CSV: x, y[, ...ignored]) once at
startup, precomputes its arc length, and tracks it with a heading +
lateral-error + damping control law (ported from a JAX bicycle-model
nominal controller).

Unlike a classic geometric pure pursuit, this controller outputs
STEERING RATE (rad/s) and ACCELERATION (m/s^2) rather than a steering
angle and speed directly -- matching a dynamic bicycle model where
steering angle (delta) and speed (v) are integrated states. Since pose
doesn't measure delta, this node tracks its own estimate of it by
integrating the commanded steering rate over time, optionally blended
with a measured estimate recovered from the commanded servo position.

Publishes ackermann_msgs/AckermannDriveStamped on /drive, populating
both the rate-style fields (steering_angle_velocity, acceleration) and
integrated single-step fields (steering_angle, speed) for compatibility
with driver stacks that expect either convention.
"""

import csv
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64


def quaternion_to_yaw(q):
    """Extract yaw (heading) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class SE2Offset:
    """Rigid transform from OptiTrack world frame -> map/track frame.

        p_map = R(theta) @ p_opti + [dx, dy]
        yaw_map = yaw_opti + theta
    """

    def __init__(self, dx: float, dy: float, dtheta: float):
        self.dx = dx
        self.dy = dy
        self.dtheta = dtheta
        self._cos = math.cos(dtheta)
        self._sin = math.sin(dtheta)

    def apply(self, x: float, y: float, yaw: float):
        x_new = self._cos * x - self._sin * y + self.dx
        y_new = self._sin * x + self._cos * y + self.dy
        yaw_new = wrap_angle(yaw + self.dtheta)
        return x_new, y_new, yaw_new


class PoseVelocityEstimator:
    """Finite-difference forward velocity + yaw rate from pose-only input,
    EMA filtered. Outputs body-frame forward speed (signed), matching what
    Odometry.twist.twist.linear.x would have given -- not a speed magnitude."""

    def __init__(self, cutoff_hz: float = 10.0, expected_rate_hz: float = 120.0):
        dt_nominal = 1.0 / expected_rate_hz
        rc = 1.0 / (2 * math.pi * cutoff_hz)
        self._alpha = dt_nominal / (rc + dt_nominal)

        self._have_prev = False
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._prev_yaw = 0.0
        self._prev_t = 0.0

        self.vx = 0.0  # body-frame forward velocity (signed)
        self.vy = 0.0  # body-frame lateral velocity (unused by pure pursuit, kept for diagnostics)
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

        c, s = math.cos(yaw), math.sin(yaw)
        vx_body_raw = c * vx_world + s * vy_world
        vy_body_raw = -s * vx_world + c * vy_world

        a = self._alpha
        self.vx = a * vx_body_raw + (1 - a) * self.vx
        self.vy = a * vy_body_raw + (1 - a) * self.vy
        self.yaw_rate = a * yaw_rate_raw + (1 - a) * self.yaw_rate

        self._prev_x, self._prev_y, self._prev_yaw, self._prev_t = x, y, yaw, t


class PurePursuitNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ---- Parameters (tune these / override via YAML or CLI) ----
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('pose_topic', '/optitrack/pose')
        self.declare_parameter('drive_topic', '/drive')

        # SE(2) origin offset: OptiTrack world frame -> map/track frame
        self.declare_parameter('origin_offset_x', 0.0)
        self.declare_parameter('origin_offset_y', 0.0)
        self.declare_parameter('origin_offset_theta', 0.0)

        # Pose-only velocity estimation
        self.declare_parameter('optitrack_rate_hz', 120.0)
        self.declare_parameter('velocity_cutoff_hz', 10.0)

        # Control law gains (match _nominal_action defaults)
        self.declare_parameter('lookahead_distance', 1.5)   # meters, arc-length lookahead
        self.declare_parameter('v_target', 2.0)               # m/s
        self.declare_parameter('k_heading', 0.0)
        self.declare_parameter('k_lateral', 0.0)
        self.declare_parameter('k_damp', 0.0)
        self.declare_parameter('k_speed', 0.0)

        # Vehicle / actuator limits (defaults match common F1TENTH dynamics params)
        self.declare_parameter('max_steering_angle', 0.4189)  # rad, +/- s_max
        self.declare_parameter('steering_velocity_min', -3.2)  # rad/s, sv_min
        self.declare_parameter('steering_velocity_max', 3.2)   # rad/s, sv_max
        self.declare_parameter('accel_max', 9.51)               # m/s^2, a_max
        self.declare_parameter('min_speed', 0.0)
        self.declare_parameter('max_speed', 6.0)
        self.declare_parameter('wheelbase', 0.33)  # meters, kept for reference/future use

        self.declare_parameter('publish_markers', True)

        self.declare_parameter('steering_angle_to_servo_gain', -1.2135)
        self.declare_parameter('steering_angle_to_servo_offset', 0.3835123)
        self.declare_parameter('feedback_correction_weight', 0.1)  # 0=pure open-loop, 1=pure feedback

        self.declare_parameter('heading_error_deadband', 0.03)  # rad, ~1.7deg - suppress noise near zero
        self.declare_parameter('interpolate_lookahead', True)    # blend between bracketing waypoints

        self.servo_gain = float(self.get_parameter('steering_angle_to_servo_gain').value)
        self.servo_offset = float(self.get_parameter('steering_angle_to_servo_offset').value)
        self.feedback_weight = float(self.get_parameter('feedback_correction_weight').value)

        self.delta_est_measured = None  # add alongside self.delta_est = 0.0

        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value

        self.offset = SE2Offset(
            self.get_parameter('origin_offset_x').value,
            self.get_parameter('origin_offset_y').value,
            self.get_parameter('origin_offset_theta').value,
        )
        self.get_logger().info(
            f"Origin offset: dx={self.offset.dx}, dy={self.offset.dy}, "
            f"dtheta={self.offset.dtheta} (update via params once calibrated)"
        )

        self.vel_estimator = PoseVelocityEstimator(
            cutoff_hz=self.get_parameter('velocity_cutoff_hz').value,
            expected_rate_hz=self.get_parameter('optitrack_rate_hz').value,
        )

        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.v_target = float(self.get_parameter('v_target').value)
        self.k_heading = float(self.get_parameter('k_heading').value)
        self.k_lateral = float(self.get_parameter('k_lateral').value)
        self.k_damp = float(self.get_parameter('k_damp').value)
        self.k_speed = float(self.get_parameter('k_speed').value)

        self.max_steering_angle = float(self.get_parameter('max_steering_angle').value)
        self.sv_min = float(self.get_parameter('steering_velocity_min').value)
        self.sv_max = float(self.get_parameter('steering_velocity_max').value)
        self.a_max = float(self.get_parameter('accel_max').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)

        self.heading_error_deadband = float(self.get_parameter('heading_error_deadband').value)
        self.interpolate_lookahead = bool(self.get_parameter('interpolate_lookahead').value)

        self.publish_markers = bool(self.get_parameter('publish_markers').value)

        if not self.waypoints_file or not os.path.isfile(self.waypoints_file):
            self.get_logger().error(
                f"waypoints_file '{self.waypoints_file}' not found. "
                "Set the 'waypoints_file' parameter to a valid CSV path."
            )
            raise FileNotFoundError(self.waypoints_file)

        self.waypoints = self._load_waypoints(self.waypoints_file)
        self._precompute_arc_length()
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints, '
            f'total arc length {self._total_arc_length:.2f} m.'
        )

        # ---- Internal state estimates (not directly observed from pose) ----
        self.delta_est = 0.0     # tracked steering angle estimate (rad)
        self.v_est = 0.0         # tracked speed setpoint (m/s) - persists across callbacks
        self.last_time = None    # for dt computation between callbacks

        # ---- QoS: OptiTrack pose is usually best-effort/high-rate ----
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        self.pose_sub = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, qos)

        self.servo_sub = self.create_subscription(
            Float64, "/sensors/servo_position_command", self.servo_callback, qos)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)

        if self.publish_markers:
            self.marker_pub = self.create_publisher(Marker, '/pure_pursuit/target_point', 10)

        self.get_logger().info(
            f'Pure pursuit node started. Listening on {self.pose_topic} (OptiTrack), '
            f'publishing to {self.drive_topic}.'
        )

    def _load_waypoints(self, path):
        """
        Load a CSV of waypoints. Expects columns: x, y[, ...extra columns ignored].
        Skips a header row automatically if it isn't numeric.
        Accepts comma or whitespace-delimited files.
        """
        pts = []
        with open(path, 'r') as f:
            sample = f.readline()
            f.seek(0)
            delimiter = ',' if ',' in sample else None
            reader = csv.reader(f, delimiter=delimiter) if delimiter else \
                (line.split() for line in f)

            for row in reader:
                row = [c for c in row if c != '']
                if len(row) < 2:
                    continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                except ValueError:
                    continue  # header row
                pts.append((x, y))

        if len(pts) == 0:
            raise ValueError(f'No valid waypoints parsed from {path}')
        return np.array(pts)

    def _precompute_arc_length(self):
        """
        Precompute per-segment vectors/lengths and cumulative arc length,
        assuming the waypoints form a closed loop (standard for an F1TENTH
        centerline). This mirrors the JAX controller's self._centerline_xy /
        self._arc_lengths_jax / self._total_arc_length.
        """
        wp = self.waypoints
        wp_next = np.roll(wp, -1, axis=0)  # wraps last point back to first
        self._seg_vec = wp_next - wp                       # (N, 2)
        self._seg_len = np.linalg.norm(self._seg_vec, axis=1)  # (N,)
        self._cum_arc = np.concatenate(([0.0], np.cumsum(self._seg_len)[:-1]))  # arc at each wp
        self._total_arc_length = float(np.sum(self._seg_len))
        self._centerline_xy = wp

    def _arc_len_at(self, pos):
        """
        Project pos onto the centerline. Returns (arc_here, idx_here, lateral_err).
        lateral_err is signed: positive when pos is to the left of the path
        direction, negative when to the right.
        """
        eps = 1e-9
        rel = pos[None, :] - self._centerline_xy               # (N, 2)
        t = np.sum(rel * self._seg_vec, axis=1) / (self._seg_len ** 2 + eps)
        t = np.clip(t, 0.0, 1.0)
        proj = self._centerline_xy + t[:, None] * self._seg_vec
        diff = pos[None, :] - proj
        dist = np.linalg.norm(diff, axis=1)
        idx_here = int(np.argmin(dist))

        arc_here = self._cum_arc[idx_here] + t[idx_here] * self._seg_len[idx_here]
        seg_dir = self._seg_vec[idx_here] / (self._seg_len[idx_here] + eps)
        rel_vec = pos - proj[idx_here]
        lateral_err = seg_dir[0] * rel_vec[1] - seg_dir[1] * rel_vec[0]
        return arc_here, idx_here, lateral_err

    def _nominal_action(self, agent_state):
        """
        Pure-pursuit-style nominal controller: arc-length lookahead point,
        heading error, signed lateral error, and yaw-rate damping produce a
        desired steering RATE; a proportional speed term produces desired
        ACCELERATION. Ported from the JAX reference implementation.
        """
        x, y, delta, v, psi, psi_dot = agent_state

        pos = np.array([x, y])
        arc_here, idx_here, lateral_err = self._arc_len_at(pos)

        lookahead_arc = math.fmod(arc_here + self.lookahead_distance, self._total_arc_length)
        if lookahead_arc < 0.0:
            lookahead_arc += self._total_arc_length

        if self.interpolate_lookahead:
            # Find the segment that brackets lookahead_arc (cum_arc is sorted ascending)
            idx_hi = int(np.searchsorted(self._cum_arc, lookahead_arc, side='right'))
            idx_lo = (idx_hi - 1) % len(self._cum_arc)
            idx_hi = idx_hi % len(self._cum_arc)

            seg_start_arc = self._cum_arc[idx_lo]
            seg_len = self._seg_len[idx_lo]
            if seg_len > 1e-9:
                frac = (lookahead_arc - seg_start_arc) / seg_len
                # handle wraparound segment where lookahead_arc < seg_start_arc numerically
                if frac < 0.0:
                    frac += self._total_arc_length / seg_len
                frac = float(np.clip(frac, 0.0, 1.0))
            else:
                frac = 0.0
            lookahead_pt = (1.0 - frac) * self._centerline_xy[idx_lo] + frac * self._centerline_xy[idx_hi]
        else:
            arc_diffs = np.abs(self._cum_arc - lookahead_arc)
            lookahead_idx = int(np.argmin(arc_diffs))
            lookahead_pt = self._centerline_xy[lookahead_idx]

        to_target = lookahead_pt - pos
        target_heading = math.atan2(to_target[1], to_target[0])
        heading_error = math.atan2(math.sin(target_heading - psi), math.cos(target_heading - psi))

        # Suppress noise-driven integration when error is within deadband
        if abs(heading_error) < self.heading_error_deadband:
            heading_error = 0.0

        desired_steering_rate = (
            self.k_heading * heading_error - self.k_lateral * lateral_err - self.k_damp * psi_dot
        )
        desired_accel = self.k_speed * (self.v_target - v)

        steering_cmd = float(np.clip(desired_steering_rate, self.sv_min, self.sv_max))
        accel_cmd = float(np.clip(desired_accel, -self.a_max, self.a_max))
        return steering_cmd, accel_cmd, lookahead_pt

    def _get_stamp_seconds(self, msg):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if t <= 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        return t

    def servo_callback(self, msg: Float64):
        # msg.data is the actual commanded servo position in servo units [0,1],
        # i.e. what ackermann_to_vesc_node produced from steering_angle. Invert
        # that same transform to recover radians.
        servo_value = msg.data
        self.delta_est_measured = (servo_value - self.servo_offset) / self.servo_gain

    def pose_callback(self, msg: PoseStamped):
        x_raw = msg.pose.position.x
        y_raw = msg.pose.position.y
        yaw_raw = quaternion_to_yaw(msg.pose.orientation)

        # Apply SE(2) offset before anything else touches the pose, so
        # velocity estimation and arc-length projection are done
        # consistently in map/track frame.
        car_x, car_y, psi = self.offset.apply(x_raw, y_raw, yaw_raw)

        now = self._get_stamp_seconds(msg)

        # Feed the raw (offset-corrected) pose + timestamp to the finite-
        # difference velocity estimator, which keeps its own internal dt.
        self.vel_estimator.update(car_x, car_y, psi, now)
        v = self.vel_estimator.vx        # signed body-frame forward speed
        psi_dot = self.vel_estimator.yaw_rate

        # Use actual elapsed time between messages for state integration,
        # rather than assuming a fixed nominal rate, guarding against
        # clock jumps / the first callback.
        dt = 0.02 if self.last_time is None else (now - self.last_time)
        dt = float(np.clip(dt, 1e-4, 0.5))
        self.last_time = now

        agent_state = (car_x, car_y, self.delta_est, v, psi, psi_dot)
        steering_rate_cmd, accel_cmd, lookahead_pt = self._nominal_action(agent_state)

        # Integrate our own steering-angle estimate since pose doesn't measure it
        delta_integrated = float(np.clip(
            self.delta_est + steering_rate_cmd * dt,
            -self.max_steering_angle, self.max_steering_angle
        ))
        if self.delta_est_measured is not None:
            self.delta_est = (
                (1.0 - self.feedback_weight) * delta_integrated
                + self.feedback_weight * self.delta_est_measured
            )
        else:
            self.delta_est = delta_integrated

        # Integrate our own speed setpoint the same way we integrate steering:
        # start from the *previously commanded* setpoint, not from the raw
        # measured velocity, so the command can actually ramp up over time
        # instead of being re-derived from (possibly stalled) real velocity
        # every callback.
        self.v_est = float(np.clip(
            self.v_est + accel_cmd * dt, self.min_speed, self.max_speed
        ))
        v_cmd = self.v_est

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        # Rate-style outputs (what the control law actually computes)
        drive_msg.drive.steering_angle_velocity = steering_rate_cmd
        drive_msg.drive.acceleration = accel_cmd
        # Integrated single-step outputs, for stacks that expect angle/speed directly
        drive_msg.drive.steering_angle = self.delta_est
        drive_msg.drive.speed = v_cmd
        self.drive_pub.publish(drive_msg)

        self.get_logger().info(
            f"x: {agent_state[0]}\ny:{agent_state[1]}\ndelta: {agent_state[2]}\n"
            f"v: {agent_state[3]}\nyaw: {agent_state[4]}\nyaw_rate: {agent_state[5]}\n"
            f"steering_angle_rate: {steering_rate_cmd}\nacceleration: {accel_cmd}"
        )

        if self.publish_markers:
            self._publish_target_marker(lookahead_pt[0], lookahead_pt[1])

    def _publish_target_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'pure_pursuit'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()