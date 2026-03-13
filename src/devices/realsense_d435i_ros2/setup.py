from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'realsense_d435i_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 自动查找launch目录下的所有launch文件
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
        # 自动查找config目录下的所有配置文件
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yipeng',
    maintainer_email='gouyipeng24@gmail.com',
    description='D435i RealSense camera package with throttling and compression',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'throttle_compress_node = realsense_d435i_ros2.ThrottleCompress:main',
        ],
    },
)