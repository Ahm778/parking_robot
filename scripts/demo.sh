#!/usr/bin/env bash
# Démo vidéo en une commande :   bash ~/parking_ws/src/parking_robot/scripts/demo.sh [graine] [méthode]
#   graine  : choisit le parking (voitures, plaques, voiture cible). Défaut 3.
#   méthode : hybrid (défaut), classic ou drl.
set -e
SEED=${1:-3}; METHOD=${2:-hybrid}
source /opt/ros/humble/setup.bash
[ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi PARKING_DEMO=1
pkill -9 gzserver 2>/dev/null || true; pkill -9 gzclient 2>/dev/null || true
pkill -9 -f "ros2 launch parking_robot" 2>/dev/null || true; sleep 1
cd ~/parking_ws/src/parking_robot
python3 scripts/make_demo_robot.py
python3 scripts/generate_world.py --seed "$SEED" --occupancy 0.75
python3 scripts/generate_map.py
python3 scripts/make_nav2_params.py
cd ~/parking_ws && colcon build --symlink-install --packages-select parking_robot && source install/setup.bash
echo "=== Gazebo s'ouvre ; Nav2 + RViz après 30 s ; la mission démarre après 60 s ==="
ros2 launch parking_robot demo.launch.py method:="$METHOD"
