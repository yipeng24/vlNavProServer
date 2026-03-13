import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

import yaml

def generate_launch_description():
    ld = LaunchDescription()

    params_file = os.path.join(get_package_share_directory('kobuki_joyop'), 'config', 'teleop_config.yaml')
    
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)['teleop_twist_joy_node']['ros__parameters']
        
    joystickNode=Node(
            package='joy',
            namespace='',
            executable='joy_node',
            name='joy_node',
            parameters=[{'dev': '/dev/input/js0', 'deadzone': 0.3, 'autorepeat_rate': 20.0}],
            arguments=['--ros-args', '--log-level', 'info']
    )

    teleopJoystickNode=Node(
            package='teleop_twist_joy',
            namespace='',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            parameters=[params],
            remappings=[('/cmd_vel', '/mux/input/joystick')],
            arguments=['--ros-args', '--log-level', 'info']
    )

    ld.add_action(joystickNode)
    ld.add_action(teleopJoystickNode)
    
    return ld
