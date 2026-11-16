"""Navigation Nav2 dans le parking : carte connue, TF fixe map -> odom (pose de départ connue), A* + DWB, RViz.

Prérequis : maps/parking.yaml (scripts/generate_map.py) et config/nav2_params.yaml (scripts/make_nav2_params.py).
Argument : rviz:=false pour lancer sans RViz.
Nav2 récent (Lyrical et plus) : navigation_launch.py ne démarre plus de gestionnaire de cycle de vie ; il expose
get_lifecycle_nodes(context) et c'est au fichier qui l'inclut de lancer un gestionnaire unique (lifecycle_manager_nav2).
Ce fichier le détecte : ancien Nav2 (Humble) -> comportement d'origine ; nouveau Nav2 -> un seul gestionnaire pour
map_server et tous les serveurs de navigation.
RViz s'ouvre avec la vue Nav2 de votre version, plus deux affichages : les places du parking (/parking/markers)
et l'image annotée de la caméra (/vision/debug).
"""
import importlib.util
import os
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from parking_robot.geometry import ROBOT_START


def parking_rviz_config(nav2):
    """Copie la configuration RViz de Nav2 et y ajoute les affichages du projet."""
    with open(os.path.join(nav2, 'rviz', 'nav2_default_view.rviz')) as f:
        cfg = yaml.safe_load(f)
    displays = cfg['Visualization Manager']['Displays']
    displays.append({'Class': 'rviz_default_plugins/MarkerArray', 'Enabled': True, 'Name': 'Places du parking',
                     'Namespaces': {}, 'Value': True,
                     'Topic': {'Depth': 5, 'Durability Policy': 'Volatile', 'History Policy': 'Keep Last',
                               'Reliability Policy': 'Reliable', 'Value': '/parking/markers'}})
    displays.append({'Class': 'rviz_default_plugins/Image', 'Enabled': True, 'Name': 'Camera (detections)',
                     'Value': True, 'Normalize Range': True, 'Max Value': 1, 'Min Value': 0, 'Median window': 5,
                     'Topic': {'Depth': 1, 'Durability Policy': 'Volatile', 'History Policy': 'Keep Last',
                               'Reliability Policy': 'Best Effort', 'Value': '/vision/debug'}})
    out = os.path.join(tempfile.gettempdir(), 'parking_nav2.rviz')
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out


def lifecycle_managers(context, sim, nav_launch):
    """Ancien Nav2 : un gestionnaire pour map_server seul (navigation_launch.py a le sien).
    Nav2 Lyrical et plus : un gestionnaire unique, lifecycle_manager_nav2, pour map_server + la navigation."""
    getter = nav2_lifecycle_nodes_getter(nav_launch)
    if getter is None:
        return [Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_map',
                     output='screen', parameters=[sim, {'autostart': True, 'node_names': ['map_server']}])]
    names = ['map_server'] + list(getter(context))
    return [Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_nav2',
                 output='screen', parameters=[sim, {'autostart': True, 'node_names': names}])]


def nav2_lifecycle_nodes_getter(nav_launch):
    """Renvoie get_lifecycle_nodes du navigation_launch.py installé, ou None (ancien Nav2 avec son propre gestionnaire)."""
    spec = importlib.util.spec_from_file_location('nav2_navigation_launch', nav_launch)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, 'get_lifecycle_nodes', None)


def generate_launch_description():
    pkg = get_package_share_directory('parking_robot')
    nav2 = get_package_share_directory('nav2_bringup')
    sim = {'use_sim_time': True}
    nav_launch = os.path.join(nav2, 'launch', 'navigation_launch.py')
    odom_x, odom_y = (0.0, 0.0) if os.environ.get('ROS_DISTRO') == 'humble' else ROBOT_START
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
             parameters=[sim, {'yaml_filename': os.path.join(pkg, 'maps', 'parking.yaml')}]),
        OpaqueFunction(function=lifecycle_managers, args=[sim, nav_launch]),
        # map -> odom. Gazebo Classic (Humble) : l'odométrie du TurtleBot3 est donnée dans le repère du monde
        # (le robot démarre déjà à ROBOT_START dans /odom), donc map = odom. Gazebo moderne : l'odométrie part
        # de (0, 0) au point de départ, donc map -> odom = ROBOT_START.
        Node(package='tf2_ros', executable='static_transform_publisher', name='map_to_odom',
             parameters=[sim],
             arguments=['--x', str(odom_x), '--y', str(odom_y), '--z', '0',
                        '--frame-id', 'map', '--child-frame-id', 'odom']),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_launch),
            launch_arguments={'use_sim_time': 'True',
                              'params_file': os.path.join(pkg, 'config', 'nav2_params.yaml')}.items()),
        Node(package='rviz2', executable='rviz2', parameters=[sim], condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', parking_rviz_config(nav2)]),
    ])
