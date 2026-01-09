from setuptools import setup

package_name = 'xbot_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/xbot_teleop.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yipeng',
    maintainer_email='you@example.com',
    description='Xbot joystick teleop to /cmd_vel',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 这里的格式：命令名 = 包名.模块名:main函数
            'xbot_teleop_node = xbot_teleop.xbot_teleop_node:main',
        ],
    },
)
