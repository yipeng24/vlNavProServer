import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():

  kobuki_keyop_node = Node(
    package='kobuki_keyop',
    executable='kobuki_keyop_node',
    name='kobuki_keyop',
    prefix='xterm -e',
    output='screen'
  )

  ld = LaunchDescription()

  ld.add_action(kobuki_keyop_node)

  return ld