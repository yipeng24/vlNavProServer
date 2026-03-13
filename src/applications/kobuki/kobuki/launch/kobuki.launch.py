import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():

    ## Start the kobuki node to establish connectivity with robot

    kobuki_node_launch    = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('kobuki_node'), 'launch', 'kobuki_node_mux.launch.py'))
    )

    ## Start the joystick keyop to establish connectivity with joystick controller

    kobuki_joyop_launch   = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('kobuki_joyop'), 'launch', 'kobuki_joyop_mux.launch.py'))
    )

    ## Start the cmd_vel_mux

    cmd_vel_mux_launch   = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('cmd_vel_mux'), 'launch', 'cmd_vel_mux.launch.py'))
    )

    ld = LaunchDescription()

    ld.add_action(kobuki_node_launch)
    ld.add_action(kobuki_joyop_launch)
    ld.add_action(cmd_vel_mux_launch)

    return ld