#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('kobuki_nav')
    slam_params = os.path.join(pkg_share, 'config', 'slam_toolbox.yaml')

    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params]
    )
    # launch 里加一个静态 TF：base_footprint -> laser
    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_basefootprint_to_laser',
        # x y z yaw pitch roll parent child
        # 把 0.20 0.00 0.15 改成你激光雷达相对底盘中心的安装位置
        arguments=['0.0', '0.0', '0.15', '3.14', '0', '0', 'base_footprint', 'laser']
    )

    life_cycle_node = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_slam',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['slam_toolbox'],
                # 'bond_timeout': 0.0,
            }]
    )

    return LaunchDescription([
        slam_node,
        tf_node,
        life_cycle_node
    ])