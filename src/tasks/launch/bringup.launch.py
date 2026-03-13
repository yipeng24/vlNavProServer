#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

def generate_launch_description():
    basebringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('kobuki_node'),
                'launch',
                'kobuki_node.launch.py'
            ])
        )
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('sllidar_ros2'),
                'launch',
                'sllidar_c1_launch.py'
            ])
        )
    )

    camera_pkg_dir = get_package_share_directory('realsense_d435i_ros2')
    config_file = os.path.join(camera_pkg_dir, 'config', 'conf_realsense_640x480x15.yaml')

    throttle_compress_node = Node(
        package='realsense_d435i_ros2',
        executable='throttle_compress_node',
        name='throttle_compress',
        namespace='',
        output='screen'
    )

    camera_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_basefootprint_to_camera',
        arguments=[
            '0.037', '-0.002', '0.559',
            '0.0003', '0.5032', '-0.0309',
            'base_footprint', 'camera_link'
        ]
    )

    camera_mount_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_camera_mount_to_realsense',
        arguments=['0', '0', '0', '0', '0', '0', 'camera_link', 'realsense2_camera_link']
    )

    slam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('kobuki_nav'),
                'launch',
                'kobuki_slam.launch.py'
            ])
        )
    )

    nav2_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('kobuki_nav'),
                'launch',
                'nomap_nav2.launch.py'
            ])
        )
    )


    realsense_actions = []
    try:
        realsense_pkg_dir = get_package_share_directory('realsense2_camera')
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
        realsense_actions.extend([
            TimerAction(period=10.0, actions=[realsense_launch]),
            TimerAction(period=15.0, actions=[throttle_compress_node]),
        ])
    except PackageNotFoundError:
        realsense_actions.append(
            LogInfo(
                msg=(
                    "[bringup] package 'realsense2_camera' not found; "
                    "skipping RealSense camera nodes."
                )
            )
        )

    return LaunchDescription([
        basebringup_launch,
        camera_base_tf,
        camera_mount_tf,
        TimerAction(period=5.0, actions=[lidar_launch]),
        *realsense_actions,
    ])
