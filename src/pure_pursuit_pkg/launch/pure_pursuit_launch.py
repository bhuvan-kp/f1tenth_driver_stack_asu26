import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('pure_pursuit_pkg'),
        'config',
        'params.yaml'
    )

    pure_pursuit_node = Node(
        package='pure_pursuit_pkg',
        executable='pure_pursuit_node',
        name='pure_pursuit_node',
        output='screen',
        parameters=[config]
    )

    return LaunchDescription([pure_pursuit_node])
