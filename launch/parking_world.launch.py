"""Lance le parking dans Gazebo et y dépose le TurtleBot3 (modèle donné par $TURTLEBOT3_MODEL, ici waffle_pi).

Piste 22.04 (ROS_DISTRO=humble)  : Gazebo Classic via gazebo_ros
Piste 26.04 (ROS_DISTRO=lyrical) : Gazebo Jetty via ros_gz_sim (+ pont ROS <-> Gazebo)
Argument : gui:=false pour lancer sans fenêtre (campagnes d'essais).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from parking_robot.geometry import ROBOT_START


def generate_launch_description():
    pkg = get_package_share_directory('parking_robot')
    tb3 = get_package_share_directory('turtlebot3_gazebo')
    world = os.path.join(pkg, 'worlds', 'parking.world')
    models = os.path.expanduser('~/gazebo_models')        # voitures et piéton (dépôt osrf/gazebo_models)
    plates = os.path.join(models, 'parking_plates')        # plaques créées par generate_world.py
    gui = LaunchConfiguration('gui')
    x, y = str(ROBOT_START[0]), str(ROBOT_START[1])

    def inc(path, args=None):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                        launch_arguments=(args or {}).items())

    demo_sdf = os.path.join(models, 'turtlebot3_waffle_pi_demo', 'model.sdf')   # créé par make_demo_robot.py
    if os.environ.get('PARKING_DEMO') == '1' and os.environ.get('ROS_DISTRO') == 'humble' and os.path.exists(demo_sdf):
        spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                     arguments=['-entity', 'waffle_pi', '-file', demo_sdf, '-x', x, '-y', y, '-z', '0.01'])
    else:
        spawn = inc(os.path.join(tb3, 'launch', 'spawn_turtlebot3.launch.py'), {'x_pose': x, 'y_pose': y})
    robot = [
        inc(os.path.join(tb3, 'launch', 'robot_state_publisher.launch.py'), {'use_sim_time': 'true'}),
        spawn,
    ]

    if os.environ.get('ROS_DISTRO') == 'humble':          # ---------- Gazebo Classic ----------
        gz = get_package_share_directory('gazebo_ros')
        sim = [
            AppendEnvironmentVariable('GAZEBO_MODEL_PATH', models),
            AppendEnvironmentVariable('GAZEBO_MODEL_PATH', plates),
            AppendEnvironmentVariable('GAZEBO_MODEL_PATH', os.path.join(tb3, 'models')),
            inc(os.path.join(gz, 'launch', 'gzserver.launch.py'), {'world': world}),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gz, 'launch', 'gzclient.launch.py')),
                                     condition=IfCondition(gui)),
        ]
    else:                                                  # ---------- Gazebo moderne (gz sim) ----------
        ros_gz_sim = get_package_share_directory('ros_gz_sim')
        sim = [
            AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', models),
            AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', plates),
            AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3, 'models')),
            inc(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py'),
                {'gz_args': f'-r -s -v2 {world}', 'on_exit_shutdown': 'true'}),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
                                     launch_arguments={'gz_args': '-g -v2', 'on_exit_shutdown': 'true'}.items(),
                                     condition=IfCondition(gui)),
            # Pont pour piloter le piéton (plugin VelocityControl) depuis ROS
            Node(package='ros_gz_bridge', executable='parameter_bridge', name='ped_bridge', output='screen',
                 arguments=['/model/ped_1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist']),
        ]

    return LaunchDescription([DeclareLaunchArgument('gui', default_value='true')] + sim + robot)
