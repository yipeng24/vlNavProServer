from setuptools import find_packages, setup

package_name = 'tasks'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/bringup.launch.py']),
        ('share/' + package_name, ['launch/nav2_waypoint.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yipeng',
    maintainer_email='gouyipeng24@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
