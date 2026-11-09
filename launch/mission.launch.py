"""Chaîne de mission : vision + localisation du véhicule + garde piétons + (DRL) + gestionnaire de mission.

Arguments :
  color:=rouge type:=voiture       requête de l'utilisateur (EF1) ; color:=aucune pour ignorer la couleur
  plate:=187-3985                  plaque recherchée « série-numéro » (aucune par défaut = pas de lecture de plaque)
  method:=classic|drl|hybrid       méthode de navigation (EF14)
  pedestrian:=true                 fait traverser le piéton (S2, S3)
  blocker:=true                    obstacle qui bloque l'allée après 20 s (S4)
  look_y:=1.0 pause_s:=3.0 inspect_y:=1.6   réglages de l'exploration (distance d'observation, attente à chaque arrêt)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LC
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    sim = {'use_sim_time': True}
    return LaunchDescription([
        DeclareLaunchArgument('color', default_value='rouge'),
        DeclareLaunchArgument('type', default_value='voiture'),
        DeclareLaunchArgument('plate', default_value='aucune'),
        DeclareLaunchArgument('method', default_value='classic'),
        DeclareLaunchArgument('pedestrian', default_value='false'),
        DeclareLaunchArgument('blocker', default_value='false'),
        DeclareLaunchArgument('look_y', default_value='1.0'),
        DeclareLaunchArgument('pause_s', default_value='3.0'),
        DeclareLaunchArgument('inspect_y', default_value='1.6'),
        Node(package='parking_robot', executable='vision_node', output='screen',
             parameters=[sim, {'target_color': ParameterValue(LC('color'), value_type=str),
                               'target_type': ParameterValue(LC('type'), value_type=str),
                               'target_plate': ParameterValue(LC('plate'), value_type=str)}]),
        Node(package='parking_robot', executable='car_locator', output='screen',
             parameters=[sim, {'target_color': ParameterValue(LC('color'), value_type=str),
                               'target_plate': ParameterValue(LC('plate'), value_type=str)}]),
        Node(package='parking_robot', executable='pedestrian_guard', parameters=[sim]),
        Node(package='parking_robot', executable='drl_controller', output='screen',
             parameters=[sim, {'mode': LC('method')}],
             condition=IfCondition(PythonExpression(["'", LC('method'), "' != 'classic'"]))),
        Node(package='parking_robot', executable='pedestrian_mover', parameters=[sim],
             condition=IfCondition(LC('pedestrian'))),
        Node(package='parking_robot', executable='scenario_events', parameters=[sim],
             condition=IfCondition(LC('blocker'))),
        Node(package='parking_robot', executable='mission_manager', output='screen',
             parameters=[sim, {'method': LC('method'),
                               'look_y': ParameterValue(LC('look_y'), value_type=float),
                               'pause_s': ParameterValue(LC('pause_s'), value_type=float),
                               'inspect_y': ParameterValue(LC('inspect_y'), value_type=float)}]),
    ])
