from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # joy_node
        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            output="screen",
        ),

        # xbot_teleop_node
        Node(
            package="xbot_teleop",
            executable="xbot_teleop_node",
            name="xbot_teleop",
            output="screen",
        ),
    ])
