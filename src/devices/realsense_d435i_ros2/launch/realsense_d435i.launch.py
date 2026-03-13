from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
import os

def generate_launch_description():

    pkg_dir = get_package_share_directory('realsense_d435i_ros2')
    config_file = os.path.join(pkg_dir, 'config', 'conf_realsense_640x480x15.yaml')
    try:
        realsense_pkg_dir = get_package_share_directory('realsense2_camera')
    except PackageNotFoundError:
        return LaunchDescription([
            LogInfo(
                msg=(
                    "[realsense_d435i] package 'realsense2_camera' not found; "
                    "skipping RealSense launch."
                )
            )
        ])

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_pkg_dir, 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'camera_name': 'realsense2_camera',
            'camera_namespace': '',
            'config_file': config_file,
            'output': 'screen',
        }.items()
    )

    throttle_compress_node = Node(
        package='realsense_d435i_ros2',
        executable='throttle_compress_node',
        name='throttle_compress',
        namespace='',
        output='screen'
    )
    delayed_compress = TimerAction(
        period=2.0,
        actions=[throttle_compress_node]
    )
    return LaunchDescription([
        realsense_launch,
        delayed_compress
    ])
