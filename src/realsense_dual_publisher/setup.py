import os
from glob import glob
from setuptools import setup

package_name = 'realsense_dual_publisher'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='RGB/IR and IMU publisher nodes for the Intel RealSense D435i',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = realsense_dual_publisher.camera_node:main',
            'imu_node = realsense_dual_publisher.imu_node:main',
            'realsense_driver = realsense_dual_publisher.realsense_driver:main',
        ],
    },
)
