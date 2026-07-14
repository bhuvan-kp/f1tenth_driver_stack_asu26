"""
flag_node.py
============
Subscribes to /joy and maintains three mutually-exclusive mode flags:
  square_flag, circle_flag, cross_flag

PS4 / DS4 button mapping (ros-humble joy default):
  Index  Button
  -----  ------
    0    Cross   (×)
    1    Circle  (○)
    2    Square  (□)  ← some drivers swap 2/3; see config/joy_params.yaml
    3    Triangle

Each button press TOGGLES its flag and clears the other two so only one
mode is active at a time.  The flags are published on /joy_flags as a
custom-free std_msgs/String JSON string so other nodes can subscribe.
They are also stored as node attributes for intra-process use.
"""

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Button index constants – edit here or override via ROS parameters
# ---------------------------------------------------------------------------
CROSS_BTN   = 0
CIRCLE_BTN  = 1
SQUARE_BTN  = 2


class FlagNode(Node):
    def __init__(self):
        super().__init__('flag_node')

        # Declare parameters so they can be overridden at launch
        self.declare_parameter('cross_btn',  CROSS_BTN)
        self.declare_parameter('circle_btn', CIRCLE_BTN)
        self.declare_parameter('square_btn', SQUARE_BTN)

        self.cross_btn  = self.get_parameter('cross_btn').value
        self.circle_btn = self.get_parameter('circle_btn').value
        self.square_btn = self.get_parameter('square_btn').value

        # Mode flags (mutually exclusive)
        self.square_flag = False
        self.circle_flag = False
        self.cross_flag  = False

        # Track previous button states to detect rising edge (press, not hold)
        self._prev_buttons: list[int] = []

        self._joy_sub = self.create_subscription(
            Joy, '/joy', self._joy_callback, 10)

        self._flag_pub = self.create_publisher(String, '/joy_flags', 10)

        self.get_logger().info(
            f'flag_node ready  [cross={self.cross_btn}  '
            f'circle={self.circle_btn}  square={self.square_btn}]')

    # ------------------------------------------------------------------
    def _joy_callback(self, msg: Joy):
        buttons = list(msg.buttons)

        # Initialise prev state on first message
        if not self._prev_buttons:
            self._prev_buttons = [0] * len(buttons)

        changed = False

        # Rising-edge detection for each button
        if self._rising(buttons, self.square_btn):
            self.square_flag = not self.square_flag
            self.circle_flag = False
            self.cross_flag  = False
            changed = True
            self.get_logger().info(
                f'Square toggled → square={self.square_flag}')

        elif self._rising(buttons, self.circle_btn):
            self.circle_flag = not self.circle_flag
            self.square_flag = False
            self.cross_flag  = False
            changed = True
            self.get_logger().info(
                f'Circle toggled → circle={self.circle_flag}')

        elif self._rising(buttons, self.cross_btn):
            self.cross_flag  = not self.cross_flag
            self.square_flag = False
            self.circle_flag = False
            changed = True
            self.get_logger().info(
                f'Cross toggled  → cross={self.cross_flag}')

        self._prev_buttons = buttons

        if changed:
            self._publish_flags()

    # ------------------------------------------------------------------
    def _rising(self, buttons: list[int], idx: int) -> bool:
        """True on the frame when button idx goes from 0 → 1."""
        if idx >= len(buttons):
            return False
        prev = self._prev_buttons[idx] if idx < len(self._prev_buttons) else 0
        return prev == 0 and buttons[idx] == 1

    # ------------------------------------------------------------------
    def _publish_flags(self):
        payload = {
            'square_flag': self.square_flag,
            'circle_flag': self.circle_flag,
            'cross_flag':  self.cross_flag,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self._flag_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FlagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
