#!/usr/bin/env python3
"""Crée config/nav2_params.yaml à partir du fichier officiel de VOTRE version de Nav2, en forçant la méthode M1 :
planificateur global A* (NavFn, use_astar) et contrôleur local DWB (équivalent moderne de DWA).

Partir du fichier de sa propre distribution évite les incompatibilités entre Humble et Lyrical
(noms de plugins, nœuds ajoutés comme collision_monitor ou docking_server, etc.).
"""
import os
import sys
from pathlib import Path

import yaml

DWB = {
    'plugin': 'dwb_core::DWBLocalPlanner',
    'debug_trajectory_details': True,
    'min_vel_x': 0.0, 'min_vel_y': 0.0, 'max_vel_x': 0.22, 'max_vel_y': 0.0, 'max_vel_theta': 1.0,
    'min_speed_xy': 0.0, 'max_speed_xy': 0.22, 'min_speed_theta': 0.0,
    'acc_lim_x': 2.5, 'acc_lim_y': 0.0, 'acc_lim_theta': 3.2,
    'decel_lim_x': -2.5, 'decel_lim_y': 0.0, 'decel_lim_theta': -3.2,
    'vx_samples': 20, 'vy_samples': 5, 'vtheta_samples': 20, 'sim_time': 1.7,
    'linear_granularity': 0.05, 'angular_granularity': 0.025, 'transform_tolerance': 0.2,
    'xy_goal_tolerance': 0.25, 'trans_stopped_velocity': 0.25,
    'short_circuit_trajectory_evaluation': True, 'stateful': True,
    'critics': ['RotateToGoal', 'Oscillation', 'BaseObstacle', 'GoalAlign', 'PathAlign', 'PathDist', 'GoalDist'],
    'BaseObstacle.scale': 0.02, 'PathAlign.scale': 32.0, 'PathAlign.forward_point_distance': 0.1,
    'GoalAlign.scale': 24.0, 'GoalAlign.forward_point_distance': 0.1, 'PathDist.scale': 32.0,
    'GoalDist.scale': 24.0, 'RotateToGoal.scale': 32.0, 'RotateToGoal.slowing_factor': 5.0,
    'RotateToGoal.lookahead_time': -1.0,
}


def main():
    distro = os.environ.get('ROS_DISTRO')
    if not distro:
        sys.exit('ROS_DISTRO vide : faites d\'abord « source /opt/ros/<distro>/setup.bash »')
    src = Path(f'/opt/ros/{distro}/share/nav2_bringup/params/nav2_params.yaml')
    p = yaml.safe_load(src.read_text())

    planner = p['planner_server']['ros__parameters']
    planner['planner_plugins'] = ['GridBased']
    navfn = 'nav2_navfn_planner/NavfnPlanner' if distro == 'humble' else 'nav2_navfn_planner::NavfnPlanner'
    planner['GridBased'] = {'plugin': navfn, 'use_astar': True, 'tolerance': 0.5, 'allow_unknown': True}

    ctrl = p['controller_server']['ros__parameters']
    ctrl['controller_plugins'] = ['FollowPath']
    ctrl['FollowPath'] = DWB

    out = Path(__file__).resolve().parents[1] / 'config' / 'nav2_params.yaml'
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(p, sort_keys=False, allow_unicode=True))
    print(f'{out} écrit depuis {src} : A* (NavFn) + DWB')


if __name__ == '__main__':
    main()
