import os
from glob import glob
from setuptools import setup

package_name = 'pure_pursuit_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Pure pursuit controller for F1TENTH using odometry only',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = pure_pursuit_pkg.pure_pursuit_node:main',
            'pure_pursuit_node_optitrack = pure_pursuit_pkg.pure_pursuit_node_optitrack:main',
        ],
    },
)
