"""
drive_node.py
=============
Subscribes to:
  /joy        – sensor_msgs/Joy          (raw controller input)
  /joy_flags  – std_msgs/String          (JSON flags from flag_node)
  /odom       – nav_msgs/Odometry        (vehicle pose / velocity)

Publishes to:
  /drive      – ackermann_msgs/AckermannDriveStamped

Behaviour
---------
* R1 is the dead-man switch: /drive is published ONLY while R1 is held.
* square_flag → square path  (open-loop timed segments)
* circle_flag → circle path  (constant steering + speed)
* cross_flag  → stop / zero command

PS4 axis/button mapping (ros-humble joy defaults):
  R1 = buttons[5]
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


# ---------------------------------------------------------------------------
# Tunable constants (also exposed as ROS parameters)
# ---------------------------------------------------------------------------
R1_BTN            = 10      # R1 button index

# Square path
SQUARE_SPEED      = 1.0    # m/s forward speed
SQUARE_SIDE_TIME  = 3.0    # seconds per straight segment
SQUARE_TURN_TIME  = 1.6    # seconds per 90° turn (tune for your vehicle)
SQUARE_STEER      = 0.5    # steering angle during turn (rad)

# Circle path
CIRCLE_SPEED      = 1.0    # m/s
CIRCLE_STEER      = 0.3    # constant steering angle (rad)

# Control loop rate
CTRL_HZ           = 20.0   # Hz


class DriveNode(Node):
    def __init__(self):
        super().__init__('drive_node')

        # ---- parameters ------------------------------------------------
        self.declare_parameter('r1_btn',           R1_BTN)
        self.declare_parameter('square_speed',     SQUARE_SPEED)
        self.declare_parameter('square_side_time', SQUARE_SIDE_TIME)
        self.declare_parameter('square_turn_time', SQUARE_TURN_TIME)
        self.declare_parameter('square_steer',     SQUARE_STEER)
        self.declare_parameter('circle_speed',     CIRCLE_SPEED)
        self.declare_parameter('circle_steer',     CIRCLE_STEER)
        self.declare_parameter('ctrl_hz',          CTRL_HZ)

        self.r1_btn           = self.get_parameter('r1_btn').value
        self.square_speed     = self.get_parameter('square_speed').value
        self.square_side_time = self.get_parameter('square_side_time').value
        self.square_turn_time = self.get_parameter('square_turn_time').value
        self.square_steer     = self.get_parameter('square_steer').value
        self.circle_speed     = self.get_parameter('circle_speed').value
        self.circle_steer     = self.get_parameter('circle_steer').value
        ctrl_hz               = self.get_parameter('ctrl_hz').value

        # ---- state -------------------------------------------------------
        self.r1_pressed   = False
        self.square_flag  = False
        self.circle_flag  = False
        self.cross_flag   = False
        self.odom: Odometry | None = None

        # Square-path open-loop state machine
        # phases cycle: straight → turn → straight → turn → ... (×4)
        self._sq_phase      = 0   # 0=straight, 1=turn  (repeats × 4 corners)
        self._sq_corner     = 0   # 0-3
        self._sq_phase_start: float | None = None

        # ---- subscriptions -----------------------------------------------
        self.create_subscription(Joy,      '/joy',       self._joy_cb,   10)
        self.create_subscription(String,   '/joy_flags', self._flags_cb, 10)
        self.create_subscription(Odometry, '/odom',      self._odom_cb,  10)

        # ---- publisher ---------------------------------------------------
        self._drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        # ---- control timer -----------------------------------------------
        self.create_timer(1.0 / ctrl_hz, self._control_loop)

        self.get_logger().info('drive_node ready')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    def _joy_cb(self, msg: Joy):
        if self.r1_btn < len(msg.buttons):
            prev = self.r1_pressed
            self.r1_pressed = bool(msg.buttons[self.r1_btn])
            if self.r1_pressed and not prev:
                self.get_logger().info('R1 pressed – drive enabled')
                self._reset_square_state()
            elif not self.r1_pressed and prev:
                self.get_logger().info('R1 released – drive disabled')

    def _flags_cb(self, msg: String):
        try:
            flags = json.loads(msg.data)
            prev = (self.square_flag, self.circle_flag, self.cross_flag)
            self.square_flag = flags.get('square_flag', False)
            self.circle_flag = flags.get('circle_flag', False)
            self.cross_flag  = flags.get('cross_flag',  False)
            if (self.square_flag, self.circle_flag, self.cross_flag) != prev:
                self._reset_square_state()
        except json.JSONDecodeError:
            self.get_logger().warn('Received malformed /joy_flags message')

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------
    def _control_loop(self):
        if not self.r1_pressed:
            return  # dead-man switch not held → publish nothing

        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        if self.square_flag:
            cmd = self._square_command(cmd)
        elif self.circle_flag:
            cmd = self._circle_command(cmd)
        elif self.cross_flag:
            cmd = self._stop_command(cmd)
        else:
            return  # no mode active – publish nothing

        self._drive_pub.publish(cmd)

    # -----------------------------------------------------------------------
    # Path generators
    # -----------------------------------------------------------------------
    def _square_command(self, cmd: AckermannDriveStamped) -> AckermannDriveStamped:
        """
        Open-loop timed square: four straight segments separated by
        four 90° left turns.
        """
        now = time.monotonic()

        if self._sq_phase_start is None:
            self._sq_phase_start = now
            self._sq_phase  = 0
            self._sq_corner = 0

        elapsed = now - self._sq_phase_start

        if self._sq_phase == 0:
            # Straight segment
            if elapsed >= self.square_side_time:
                self._sq_phase       = 1
                self._sq_phase_start = now
                elapsed              = 0.0
            cmd.drive.speed           = self.square_speed
            cmd.drive.steering_angle  = 0.0

        if self._sq_phase == 1:
            # Turn segment
            if elapsed >= self.square_turn_time:
                self._sq_corner += 1
                if self._sq_corner >= 4:
                    self._sq_corner = 0  # loop the square
                self._sq_phase       = 0
                self._sq_phase_start = now
            cmd.drive.speed           = self.square_speed
            cmd.drive.steering_angle  = self.square_steer  # left turn

        return cmd

    def _circle_command(self, cmd: AckermannDriveStamped) -> AckermannDriveStamped:
        """Constant speed + constant steering → circular arc."""
        cmd.drive.speed          = self.circle_speed
        cmd.drive.steering_angle = self.circle_steer
        return cmd

    def _stop_command(self, cmd: AckermannDriveStamped) -> AckermannDriveStamped:
        """Cross flag → zero speed, straight steering."""
        cmd.drive.speed          = 0.0
        cmd.drive.steering_angle = 0.0
        return cmd

    # -----------------------------------------------------------------------
    def _reset_square_state(self):
        self._sq_phase       = 0
        self._sq_corner      = 0
        self._sq_phase_start = None


def main(args=None):
    rclpy.init(args=args)
    node = DriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
