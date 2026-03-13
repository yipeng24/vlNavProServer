from setuptools import find_packages, setup

package_name = 'container'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/container.launch.py']),
    ],
    install_requires=[
        'setuptools',
        'openai',
        'python-dotenv',
    ],
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
            'container_node = container.singleTh_container:main',
            "waypoint_client = container.waypoint_client:main",
            "bridge_node = container.bridge_node:main",
        ],
    },
)
