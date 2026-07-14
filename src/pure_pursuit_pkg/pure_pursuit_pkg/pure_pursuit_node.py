#!/usr/bin/env python3
"""
Pure pursuit controller for F1TENTH.

Subscribes only to odometry (nav_msgs/Odometry). Loads a centerline
waypoint file (CSV: x, y[, ...ignored]) once at startup, precomputes
its arc length, and tracks it with a heading + lateral-error + damping
control law (ported from a JAX bicycle-model nominal controller).

Unlike a classic geometric pure pursuit, this controller outputs
STEERING RATE (rad/s) and ACCELERATION (m/s^2) rather than a steering
angle and speed directly -- matching a dynamic bicycle model where
steering angle (delta) and speed (v) are integrated states. Since
odometry doesn't measure delta, this node tracks its own estimate of
it by integrating the commanded steering rate over time.

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

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64


def quaternion_to_yaw(q):
    """Extract yaw (heading) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuitNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ---- Parameters (tune these / override via YAML or CLI) ----
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('drive_topic', '/drive')

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
        self.odom_topic = self.get_parameter('odom_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value

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

        # ---- Internal state estimates (not directly observed from odometry) ----
        self.delta_est = 0.0     # tracked steering angle estimate (rad)
        self.last_time = None    # for dt computation between callbacks
        self.v_est = 0.0

        # ---- QoS: odometry is usually best-effort/high-rate ----
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, qos)

        self.servo_sub = self.create_subscription(
            Float64, "/sensors/servo_position_command", self.servo_callback, qos)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)

        if self.publish_markers:
            self.marker_pub = self.create_publisher(Marker, '/pure_pursuit/target_point', 10)

        self.get_logger().info(
            f'Pure pursuit node started. Listening on {self.odom_topic}, '
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

    def odom_callback(self, msg: Odometry):
        car_x = msg.pose.pose.position.x
        car_y = msg.pose.pose.position.y
        psi = quaternion_to_yaw(msg.pose.pose.orientation)
        v = msg.twist.twist.linear.x        # forward speed, body frame
        psi_dot = msg.twist.twist.angular.z  # yaw rate

        now = self._get_stamp_seconds(msg)
        dt = 0.02 if self.last_time is None else (now - self.last_time)
        dt = float(np.clip(dt, 1e-4, 0.5))  # guard against clock jumps/first callback
        self.last_time = now

        agent_state = (car_x, car_y, self.delta_est, v, psi, psi_dot)
        steering_rate_cmd, accel_cmd, lookahead_pt = self._nominal_action(agent_state)

        # Integrate our own steering-angle estimate since odometry doesn't measure it
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
