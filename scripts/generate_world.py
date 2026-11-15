#!/usr/bin/env python3
"""Génère worlds/parking.world et config/parking_slots.yaml à partir d'une graine (essais reproductibles).

Usage (depuis le dossier du package) :
  python3 scripts/generate_world.py --seed 1
  python3 scripts/generate_world.py --seed 7 --occupancy 0.8 --same-color-decoys 2   # scénario S6
  python3 scripts/generate_world.py --seed 2 --target-plate 187-3985                # plaque imposée
Chaque voiture reçoit une plaque tunisienne « série تونس numéro », écrite « série-numéro » dans les commandes
(modèles créés dans ~/gazebo_models/parking_plates).
Le simulateur est déduit de $ROS_DISTRO (humble -> Gazebo Classic, sinon Gazebo moderne) ; --sim force le choix.
"""
import argparse
import math
import os
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking_robot.geometry import (AISLE_HALF, BLOCKER_PARK, DEMO, MEDIANS, PED, ROWS, WALL_X, WALL_Y, CAR_CY, CAR_FRONT, ROBOT_START, ROW_SIGN, SLOT_L, SLOT_W,  # noqa: E402
                                    STOP_CY, STOP_SIZE, WALL_C, WALL_H, WALL_T, XS, slots)
from parking_robot.plates import PLATE_Z, display, random_plate, write_plate_model  # noqa: E402

DECOYS = ['hatchback_blue', 'hatchback', 'suv', 'pickup']      # voitures « autres » (pas la cible)
SAME_COLOR_DECOY = 'suv_red'                                    # créé par scripts/make_decoy.py (scénario S6)


def box(name, x, y, sx, sy, sz, rgba, collide=True, static=True, z=None):
    geom = f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    col = f"<collision name='c'>{geom}</collision>" if collide else ''
    z = sz / 2 if z is None else z
    return (f"<model name='{name}'><static>{str(static).lower()}</static><pose>{x} {y} {z} 0 0 0</pose>"
            f"<link name='l'>{col}<visual name='v'>{geom}<material><ambient>{rgba}</ambient>"
            f"<diffuse>{rgba}</diffuse></material></visual></link></model>")


def pedestrian(sim, x, y):
    """Piéton : collision cylindrique (vue par le lidar) + maillage person_standing (vu par la caméra)."""
    common = ("<collision name='c'><pose>0 0 0.85 0 0 0</pose><geometry><cylinder><radius>0.25</radius>"
              "<length>1.7</length></cylinder></geometry>{surface}</collision>"
              "<visual name='v'><pose>0 0 0.02 0.04 0 0</pose><geometry><mesh>"
              "<uri>model://person_standing/meshes/standing.dae</uri></mesh></geometry></visual>")
    if sim == 'classic':     # statique, téléporté par pedestrian_mover via /gazebo/set_entity_state
        return (f"<model name='ped_1'><static>true</static><pose>{x} {y} 0 0 0 0</pose>"
                f"<link name='l'>{common.format(surface='')}</link></model>")
    # Gazebo moderne : modèle dynamique sans frottement, piloté en vitesse par VelocityControl
    surface = "<surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface>"
    return (f"<model name='ped_1'><pose>{x} {y} 0.01 0 0 0</pose><link name='l'>"
            "<inertial><pose>0 0 0.6 0 0 0</pose><mass>70</mass><inertia><ixx>17</ixx><iyy>17</iyy>"
            "<izz>2</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>"
            f"{common.format(surface=surface)}</link>"
            "<plugin filename='gz-sim-velocity-control-system' name='gz::sim::systems::VelocityControl'/>"
            "</model>")


GZ_SYSTEMS = """
<plugin filename='gz-sim-physics-system' name='gz::sim::systems::Physics'/>
<plugin filename='gz-sim-user-commands-system' name='gz::sim::systems::UserCommands'/>
<plugin filename='gz-sim-scene-broadcaster-system' name='gz::sim::systems::SceneBroadcaster'/>
<plugin filename='gz-sim-sensors-system' name='gz::sim::systems::Sensors'><render_engine>ogre2</render_engine></plugin>
<plugin filename='gz-sim-imu-system' name='gz::sim::systems::Imu'/>"""

CLASSIC_SYSTEMS = """
<plugin name='gazebo_ros_state' filename='libgazebo_ros_state.so'><ros><namespace>/gazebo</namespace></ros><update_rate>20</update_rate></plugin>"""

COMMON_SCENE = """
<physics type='ode'><max_step_size>0.001</max_step_size><real_time_factor>1</real_time_factor><real_time_update_rate>1000</real_time_update_rate></physics>
<light type='directional' name='sun'><cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose>
<diffuse>{light} {light} {light} 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.5 0.1 -0.9</direction></light>
<model name='ground_plane'><static>true</static><link name='link'>
<collision name='collision'><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
<visual name='visual'><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
<material><ambient>0.35 0.35 0.35 1</ambient><diffuse>0.35 0.35 0.35 1</diffuse></material></visual></link></model>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--occupancy', type=float, default=0.7, help='part des places occupées (0-1)')
    ap.add_argument('--target', default='hatchback_red', help='modèle de la voiture recherchée')
    ap.add_argument('--same-color-decoys', type=int, default=0, help='leurres de même couleur (S6)')
    ap.add_argument('--target-far', action='store_true', help='cible dans la colonne la plus éloignée (S5)')
    ap.add_argument('--light', type=float, default=0.8, help='intensité du soleil (ENF7 : 0.4, 0.8, 1.0)')
    ap.add_argument('--target-plate', default='', help='plaque de la voiture cible, « série-numéro », ex. 187-3985 (sinon au hasard)')
    ap.add_argument('--plates-dir', default=os.path.expanduser('~/gazebo_models/parking_plates'))
    ap.add_argument('--sim', choices=['auto', 'classic', 'gz'], default='auto')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    sim = a.sim if a.sim != 'auto' else ('classic' if os.environ.get('ROS_DISTRO') == 'humble' else 'gz')
    rng = random.Random(a.seed)

    parts = []
    cx, cy, t = WALL_X, WALL_Y, WALL_T
    for i, (x, y, sx, sy) in enumerate([(0, cy, 2 * cx + t, t), (0, -cy, 2 * cx + t, t), (cx, 0, t, 2 * cy), (-cx, 0, t, 2 * cy)]):
        parts.append(box(f'wall_{i}', x, y, sx, sy, WALL_H, '0.6 0.6 0.6 1'))
    for row, s in ROW_SIGN.items():
        for k in range(len(XS) + 1):                                  # lignes blanches (sans collision)
            x = XS[0] - SLOT_W / 2 + k * SLOT_W
            parts.append(box(f'line_{row}{k}', x, ROWS[row]['slot_cy'], 0.1, SLOT_L, 0.005, '1 1 1 1', collide=False))
        for i, x in enumerate(XS):                                    # butées (vues par le lidar)
            parts.append(box(f'stop_{row}{i + 1}', x, ROWS[row]['stop_y'], *STOP_SIZE, '0.9 0.8 0.1 1'))

    all_slots = slots()
    n_occ = max(1, round(a.occupancy * len(all_slots)))
    if a.target_far:
        far = [s for s in all_slots if s['x'] == max(XS)]
        target = rng.choice(far)
        occupied = [target] + rng.sample([s for s in all_slots if s is not target], n_occ - 1)
    else:
        occupied = rng.sample(all_slots, n_occ)
        target = rng.choice(occupied)
    others = [s for s in occupied if s is not target]
    same = rng.sample(others, min(a.same_color_decoys, len(others)))

    out_slots, used = [], set()
    for s in all_slots:
        model = None
        if s is target:
            model = a.target
        elif any(s is o for o in same):
            model = SAME_COLOR_DECOY
        elif any(s is o for o in occupied):
            model = rng.choice(DECOYS)
        plate = None
        if model:
            plate = a.target_plate if (s is target and a.target_plate) else random_plate(rng)
            while plate in used:
                plate = random_plate(rng)
            used.add(plate)
            sign = ROW_SIGN[s['row']]
            yaw = -math.pi / 2 if sign > 0 else math.pi / 2        # l'avant (+x du modèle) regarde l'allée
            parts.append(f"<include><name>car_{s['id']}</name><uri>model://{model}</uri><static>true</static>"
                         f"<pose>{s['x']} {ROWS[s['row']]['car_cy']} 0 0 0 {yaw:.4f}</pose></include>")
            # plaque : rectangle texturé posé juste devant le pare-chocs, face à l'allée
            name = f"plate_{s['id']}"
            write_plate_model(a.plates_dir, name, plate)
            py = ROWS[s['row']]['car_cy'] - sign * (CAR_FRONT.get(model, 2.5) + 0.02)
            parts.append(f"<include><name>{name}</name><uri>model://{name}</uri><static>true</static>"
                         f"<pose>{s['x']} {py:.3f} {PLATE_Z} 0 0 {yaw:.4f}</pose></include>")
        out_slots.append({**s, 'gt_car': model, 'gt_plate': plate})

    parts.append(pedestrian(sim, PED[0], PED[1]))     # entre deux colonnes d'observation (x = -3 et x = 0)
    parts.append(box('blocker', BLOCKER_PARK[0], BLOCKER_PARK[1], 0.8, 3.0, 1.0, ('1 0.45 0 1' if DEMO else '0.8 0.2 0.2 1')))   # S4 : rangé hors du parking

    for k, (mx, my, sx, sy) in enumerate(MEDIANS):                # terre-plein (mode démo)
        parts.append(box(f'median_{k}', mx, my, sx, sy, 0.2, '0.55 0.55 0.5 1'))
    if DEMO:                                                      # décor de la vidéo
        from parking_robot.demo_decor import decor
        parts += decor()
    systems = CLASSIC_SYSTEMS if sim == 'classic' else GZ_SYSTEMS
    world = ("<?xml version='1.0'?>\n<sdf version='1.7'><world name='parking'>" + systems
             + COMMON_SCENE.format(light=a.light) + '\n' + '\n'.join(parts) + '\n</world></sdf>\n')
    out = Path(a.out)
    (out / 'worlds').mkdir(parents=True, exist_ok=True)
    (out / 'config').mkdir(parents=True, exist_ok=True)
    (out / 'worlds' / 'parking.world').write_text(world)
    with open(out / 'config' / 'parking_slots.yaml', 'w') as f:
        target_plate = next(o['gt_plate'] for o in out_slots if o['id'] == target['id'])
        yaml.safe_dump({'seed': a.seed, 'sim': sim, 'target_model': a.target, 'gt_target_slot': target['id'],
                        'gt_target_plate': target_plate,
                        'robot_start': list(ROBOT_START), 'slots': out_slots}, f, sort_keys=False, allow_unicode=True)
    print(f"[{sim}] monde généré (graine {a.seed}) : cible {a.target} en {target['id']}, "
          f"plaque {display(target_plate)} -> à demander avec plate:={target_plate}")


if __name__ == '__main__':
    main()
