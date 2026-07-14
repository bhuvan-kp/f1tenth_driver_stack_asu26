from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # NOTE: camera_node and imu_node used to be launched as two separate
    # processes, each opening its own rs.pipeline(). On Jetson (where
    # librealsense is built with FORCE_RSUSB_BACKEND=true), the RSUSB
    # backend claims the whole USB device exclusively per-process, so the
    # second process to start would fail with "No device connected".
    #
    # realsense_driver runs both node objects in a single process/executable
    # backed by one shared rs.pipeline(), which is what the RSUSB backend
    # actually supports. Topics and node names are unchanged.
    return LaunchDescription([
        Node(
            package='realsense_dual_publisher',
            executable='realsense_driver',
            output='screen',
        ),
    ])

