#!/usr/bin/env bash
# Démo avec fenêtre de choix du véhicule :   bash ~/parking_ws/src/parking_robot/scripts/demo_choix.sh [graine]
# Ouvre Gazebo + Nav2/RViz, puis une fenêtre listant les voitures garées : cliquez « Chercher » sur l'une d'elles.
# Fermer la fenêtre arrête tout.
set -e
SEED=${1:-3}
source /opt/ros/humble/setup.bash
[ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi PARKING_DEMO=1
if ! python3 -c "import tkinter" 2>/dev/null; then
  echo "Il manque Tkinter pour la fenêtre. Installez-le une fois :  sudo apt install -y python3-tk"; exit 1
fi
pkill -9 gzserver 2>/dev/null || true; pkill -9 gzclient 2>/dev/null || true
pkill -9 -f "ros2 launch parking_robot" 2>/dev/null || true; sleep 1
cd ~/parking_ws/src/parking_robot
python3 scripts/make_demo_robot.py
python3 scripts/generate_world.py --seed "$SEED" --occupancy 0.75
python3 scripts/generate_map.py
python3 scripts/make_nav2_params.py
cd ~/parking_ws && colcon build --symlink-install --packages-select parking_robot && source install/setup.bash
setsid ros2 launch parking_robot demo.launch.py auto_mission:=false > /tmp/demo_choix.log 2>&1 &
LAUNCH=$!
trap 'kill -INT -- -$LAUNCH 2>/dev/null; sleep 3; kill -9 -- -$LAUNCH 2>/dev/null; pkill -9 gzserver; pkill -9 gzclient; true' EXIT
echo "=== Gazebo s'ouvre ; Nav2 + RViz dans 30 s ; la fenêtre de choix s'ouvre dans 45 s ==="
sleep 45
python3 ~/parking_ws/src/parking_robot/scripts/choisir_vehicule.py
