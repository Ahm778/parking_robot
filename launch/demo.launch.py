"""Démo vidéo en UNE commande : Gazebo + Nav2/RViz + mission, dans le parking démo (PARKING_DEMO=1).

    ros2 launch parking_robot demo.launch.py                    # méthode hybride, piéton et caisse
    ros2 launch parking_robot demo.launch.py method:=classic    # ou drl
La plaque recherchée est lue dans config/parking_slots.yaml (celle du monde généré) et affichée au démarrage.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC


def generate_launch_description():
    if os.environ.get('PARKING_DEMO') != '1':
        raise RuntimeError('Faites d\'abord « export PARKING_DEMO=1 » (ou utilisez scripts/demo.sh)')
    pkg = get_package_share_directory('parking_robot')
    with open(os.path.join(pkg, 'config', 'parking_slots.yaml')) as f:
        plate = yaml.safe_load(f)['gt_target_plate']
    inc = lambda name, args: IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', name)), launch_arguments=args.items())
    return LaunchDescription([
        DeclareLaunchArgument('method', default_value='hybrid'),
        DeclareLaunchArgument('pedestrian', default_value='true'),
        DeclareLaunchArgument('blocker', default_value='true'),
        DeclareLaunchArgument('auto_mission', default_value='true'),   # false : choix dans la fenêtre
        LogInfo(msg=f'=== DÉMO : le robot va chercher la voiture immatriculée {plate} ==='),
        inc('parking_world.launch.py', {'gui': 'true'}),
        TimerAction(period=30.0, actions=[inc('navigation.launch.py', {'rviz': 'true'})]),
        TimerAction(period=60.0, condition=IfCondition(LC('auto_mission')), actions=[
            LogInfo(msg=f'=== Mission lancée : plaque {plate} ==='),
            inc('mission.launch.py', {'color': 'aucune', 'plate': plate, 'method': LC('method'),
                                      'pedestrian': LC('pedestrian'), 'blocker': LC('blocker'),
                                      'pause_s': '1.0'})]),
    ])
