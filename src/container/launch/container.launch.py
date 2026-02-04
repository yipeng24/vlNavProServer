from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    respawn = LaunchConfiguration('respawn')

    return LaunchDescription([
        DeclareLaunchArgument(
            'respawn',
            default_value='false',
            description='Whether to respawn nodes if they crash'
        ),

        # 你的 container_node
        Node(
            package='container',
            executable='container_node',
            name='container_node',
            output='screen',
            respawn=respawn,
            respawn_delay=1.0,
        ),

        # joy_node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            respawn=respawn,
            respawn_delay=1.0,
        ),
    ])
