from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ppo_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob('data/*.csv')),
        (os.path.join('share', package_name, 'models'), glob('models/*.safetensors')),
        (os.path.join('share', package_name, 'data'), glob('data/*.yaml')),
        (os.path.join('share', package_name, 'data'), glob('data/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bhuvan_k_prasad',
    maintainer_email='bhuvan_k_prasad@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "observation_node = ppo_controller.observation_node:main",
            "policy_node = ppo_controller.policy_node:main",
            "observation_node_lidar = ppo_controller.observation_node_lidar:main",
            "policy_node_lidar = ppo_controller.policy_node_lidar:main",
            #'sim_to_hardware_bridge = ppo_controller.sim_to_hardware_bridge:main',
            'observation_node_optitrack = ppo_controller.observation_node_optitrack:main',
        ],
    },
)
