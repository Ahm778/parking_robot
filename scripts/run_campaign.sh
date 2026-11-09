#!/usr/bin/env bash
# Campagne d'essais automatique (identique sur les deux pistes).
# Usage : ./scripts/run_campaign.sh S1 classic 50 [couleur|plaque]
#   4e argument : critère d'identification (couleur par défaut ; « plaque » = recherche par plaque seule)
# Résultats : une ligne par essai dans ~/results.csv (un essai sans ligne = échec par dépassement de temps).
set -u
SC=${1:?scénario S1..S6}; METHOD=${2:?classic|drl|hybrid}; N=${3:-50}; CRIT=${4:-couleur}
PKG_SRC="$(cd "$(dirname "$0")/.." && pwd)"
EXTRA=""
case $SC in
  S1) ARGS="--occupancy 0.3";;
  S2) ARGS="--occupancy 0.9"; EXTRA="pedestrian:=true";;
  S3) ARGS="--occupancy 0.6"; EXTRA="pedestrian:=true";;
  S4) ARGS="--occupancy 0.6"; EXTRA="blocker:=true";;
  S5) ARGS="--occupancy 0.9 --target-far";;
  S6) ARGS="--occupancy 0.8 --same-color-decoys 2";;
  *) echo "Scénario inconnu : $SC"; exit 1;;
esac

cleanup() {
  for p in "${PIDS[@]:-}"; do kill -INT -- -"$p" 2>/dev/null; done
  sleep 5
  for p in "${PIDS[@]:-}"; do kill -9 -- -"$p" 2>/dev/null; done
  pkill -9 -f gzserver; pkill -9 -f gzclient; pkill -9 -f "gz sim"
  sleep 3
}

for SEED in $(seq 1 "$N"); do
  echo "=== $SC / $METHOD / graine $SEED ==="
  python3 "$PKG_SRC/scripts/generate_world.py" --seed "$SEED" $ARGS
  if [ "$CRIT" = "plaque" ]; then
    PLATE=$(python3 -c "import yaml; print(yaml.safe_load(open('$PKG_SRC/config/parking_slots.yaml'))['gt_target_plate'])")
    QUERY=(color:=aucune "plate:=$PLATE")
  else
    QUERY=(color:=rouge)
  fi
  PIDS=()
  setsid ros2 launch parking_robot parking_world.launch.py gui:=false > /tmp/world.log 2>&1 & PIDS+=($!)
  sleep 20
  setsid ros2 launch parking_robot navigation.launch.py rviz:=false > /tmp/nav.log 2>&1 & PIDS+=($!)
  sleep 15
  setsid ros2 launch parking_robot mission.launch.py "${QUERY[@]}" method:="$METHOD" $EXTRA > /tmp/mission.log 2>&1 & PIDS+=($!)
  timeout 600 ros2 run parking_robot metrics_logger --ros-args -p use_sim_time:=true \
      -p scenario:="$SC" -p method:="$METHOD-$CRIT" -p seed:="$SEED" -p out:="$HOME/results.csv" \
      || echo "Graine $SEED : pas de résultat (dépassement de temps)"
  cleanup
done
