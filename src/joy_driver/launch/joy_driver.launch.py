"""
joy_driver.launch.py
====================
Launches:
  1. joy_node        – reads /dev/input/jsX and publishes /joy
  2. flag_node       – converts button presses to mode flags on /joy_flags
  3. drive_node      – generates AckermannDriveStamped on /drive
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---------- launch arguments (override from CLI) ----------
    joy_dev_arg = DeclareLaunchArgument(
        'joy_dev', default_value='/dev/input/js0',
        description='Joystick device path')

    # ---- joy_node (publishes raw /joy) ----
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'device_id': 0,
            'deadzone': 0.05,
            'autorepeat_rate': 20.0,
        }],
        output='screen',
    )

    # ---- flag_node ----
    flag_node = Node(
        package='joy_driver',
        executable='flag_node',
        name='flag_node',
        parameters=[{
            'cross_btn':  0,
            'circle_btn': 1,
            'square_btn': 2,
        }],
        output='screen',
    )

    # ---- drive_node ----
    drive_node = Node(
        package='joy_driver',
        executable='drive_node',
        name='drive_node',
        parameters=[{
            'r1_btn':           10,
            'square_speed':     1.0,
            'square_side_time': 3.0,
            'square_turn_time': 1.6,
            'square_steer':     0.5,
            'circle_speed':     1.0,
            'circle_steer':     0.3,
            'ctrl_hz':          20.0,
        }],
        output='screen',
    )

    return LaunchDescription([
        joy_dev_arg,
        joy_node,
        flag_node,
        drive_node,
    ])
