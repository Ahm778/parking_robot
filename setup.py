import os
from glob import glob
from setuptools import setup

package_name = 'parking_robot'


def files(pattern):
    return [f for f in glob(pattern) if os.path.isfile(f)]


setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), files('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), files('worlds/*')),
        (os.path.join('share', package_name, 'maps'), files('maps/*')),
        (os.path.join('share', package_name, 'config'), files('config/*.yaml')),
        (os.path.join('share', package_name, 'weights'), files('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Groupe Robot Parking',
    maintainer_email='groupe@example.com',
    description='Robot parking : vision, localisation du véhicule, navigation classique et DRL',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vision_node = parking_robot.vision_node:main',
            'car_locator = parking_robot.car_locator:main',
            'mission_manager = parking_robot.mission_manager:main',
            'pedestrian_mover = parking_robot.pedestrian_mover:main',
            'pedestrian_guard = parking_robot.pedestrian_guard:main',
            'scenario_events = parking_robot.scenario_events:main',
            'drl_controller = parking_robot.drl_controller:main',
            'metrics_logger = parking_robot.metrics_logger:main',
        ],
    },
)
