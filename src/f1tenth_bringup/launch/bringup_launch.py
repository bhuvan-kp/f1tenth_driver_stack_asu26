"""
Top-level bringup launch file for the F1TENTH car.

This package intentionally contains NO nodes of its own -- it only
includes each subsystem's existing launch file. To add a new subsystem
in the future (e.g. a sensor driver package):

  1. Add the package as an <exec_depend> in f1tenth_bringup/package.xml
  2. Add one DeclareLaunchArgument enable_<name> block below
  3. Add one IncludeLaunchDescription block in generate_launch_description()

Each subsystem can then be toggled on/off independently at launch time,
e.g.:
    ros2 launch f1tenth_bringup bringup_launch.py enable_joy_driver:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    enable_f1tenth_stack_arg = DeclareLaunchArgument(
        'enable_f1tenth_stack', default_value='true',
        description='Launch the core F1TENTH driver stack (VESC, LIDAR, etc.)',
    )
    enable_joy_driver_arg = DeclareLaunchArgument(
        'enable_joy_driver', default_value='true',
        description='Launch the joystick/teleop driver',
    )
    enable_realsense_arg = DeclareLaunchArgument(
        'enable_realsense', default_value='true',
        description='Launch the RealSense camera + IMU driver',
    )
    enable_natnet_arg = DeclareLaunchArgument(
        'enable_natnet', default_value='true',
        description='Launch the OptiTrack/Motive NatNet mocap driver',
    )
    natnet_server_ip_arg = DeclareLaunchArgument(
        'natnet_server_ip', default_value='192.168.0.217',
        description='IP of the Motive PC (NatNet server)',
    )
    natnet_client_ip_arg = DeclareLaunchArgument(
        'natnet_client_ip', default_value='192.168.0.109',
        description='IP of this Jetson (NatNet client)',
    )

    # --- Core F1TENTH driver stack -----------------------------------------
    f1tenth_stack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('f1tenth_stack'),
                'launch',
                'bringup_launch.py',
            ])
        ),
        condition=IfCondition(LaunchConfiguration('enable_f1tenth_stack')),
    )

    # --- Joystick / teleop driver -------------------------------------------
    joy_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('joy_driver'),
                'launch',
                'joy_driver.launch.py',
            ])
        ),
        condition=IfCondition(LaunchConfiguration('enable_joy_driver')),
    )

    # --- RealSense camera + IMU driver --------------------------------------
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('realsense_dual_publisher'),
                'launch',
                'realsense_dual.launch.py',
            ])
        ),
        condition=IfCondition(LaunchConfiguration('enable_realsense')),
    )
    
    # --- OptiTrack / Motive NatNet mocap driver ------------------------------
    natnet_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('natnet_ros2'),
                'launch',
                'natnet_ros2.launch.py',
            ])
        ),
        launch_arguments={
            'serverIP': LaunchConfiguration('natnet_server_ip'),
            'clientIP': LaunchConfiguration('natnet_client_ip'),
            'serverType': 'multicast',
            'pub_rigid_body': 'true',
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_natnet')),
    )

    # --- Add further future subsystems here, following the same pattern ---

    return LaunchDescription([
        enable_f1tenth_stack_arg,
        enable_joy_driver_arg,
        enable_realsense_arg,
        enable_natnet_arg,
        natnet_server_ip_arg,
        natnet_client_ip_arg,
        f1tenth_stack_launch,
        joy_driver_launch,
        realsense_launch,
        natnet_launch,
    ])
