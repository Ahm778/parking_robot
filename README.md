# parking_robot

Robot mobile (TurtleBot3 waffle_pi simulé) qui retrouve un véhicule dans un parking.
Fonctionne sur les deux pistes du groupe, sans modification du code :

| Piste | Système | ROS 2 | Simulateur | /cmd_vel |
| --- | --- | --- | --- | --- |
| A | Ubuntu 22.04 | Humble | Gazebo Classic 11 | Twist |
| B | Ubuntu 26.04 | Lyrical | Gazebo Jetty (gz-sim 10) | TwistStamped |

La piste est détectée automatiquement par `$ROS_DISTRO` (humble -> Classic, sinon Gazebo moderne).

## Démarrage rapide (après installation, voir le guide)

```bash
cd ~/parking_ws/src/parking_robot
python3 scripts/generate_world.py --seed 1      # monde + places (graine reproductible)
python3 scripts/generate_map.py                 # carte Nav2 (murs + butées)
python3 scripts/make_nav2_params.py             # paramètres Nav2 : A* + DWB
# déposer weights/yolov5n.onnx (export 320x320, voir le guide)
cd ~/parking_ws && colcon build --symlink-install && source install/setup.bash

ros2 launch parking_robot parking_world.launch.py          # terminal 1
ros2 launch parking_robot navigation.launch.py             # terminal 2
ros2 launch parking_robot mission.launch.py color:=rouge   # terminal 3
# par plaque : mission.launch.py color:=aucune plate:=187-3985  (sudo apt install tesseract-ocr)
```

## Contenu

- `parking_robot/slot_tracker.py` : état de chaque place (affichage RViz, vision active)
- `parking_robot/plates.py` : plaques tunisiennes (création dans le monde, lecture OCR Tesseract)
- `parking_robot/` : nœuds (vision_node, car_locator, mission_manager, pedestrian_mover, pedestrian_guard,
  scenario_events, drl_controller, metrics_logger) et modules partagés (geometry, common, drl_common, drl_env)
- `launch/` : parking_world, navigation, mission
- `scripts/` : generate_world, generate_map, make_nav2_params, make_decoy, train_ppo, export_policy,
  run_campaign.sh, analyze
- `weights/` : yolov5n.onnx (à ajouter), policy.npz (politique DRL exportée)
