#!/usr/bin/env bash
# Campagne d'essais automatique (identique sur les deux pistes).
# Usage : ./scripts/run_campaign.sh S1 classic 50 [couleur|plaque]
#   4e argument : critère d'identification (couleur par défaut ; « plaque » = recherche par plaque seule)
# Résultats : une ligne par essai dans ~/results.csv (un essai sans ligne = échec par dépassement de temps).
# Journaux de chaque essai gardés dans ~/logs_campagne/<scénario>_<méthode>_<graine>_{world,nav,mission}.log
# Durée max d'un essai : 900 s par défaut (TIMEOUT=600 ./scripts/run_campaign.sh ... pour changer)
# 5e argument : première graine (ex. 11 avec N=10 -> graines 11 à 20)
# Rejouer seulement certaines graines : SEEDS="4 5" ./scripts/run_campaign.sh S4 classic 5 plaque
set -u
SC=${1:?scénario S1..S6}; METHOD=${2:?classic|drl|hybrid}; N=${3:-50}; CRIT=${4:-couleur}; START=${5:-1}
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

LOGDIR="$HOME/logs_campagne"; mkdir -p "$LOGDIR"
TIMEOUT=${TIMEOUT:-900}     # durée max d'un essai (s), la même pour toutes les méthodes
for SEED in ${SEEDS:-$(seq "$START" "$((START + N - 1))")}; do
  L="$LOGDIR/${SC}_${METHOD}_${SEED}"
  echo "=== $SC / $METHOD / graine $SEED ==="
  python3 "$PKG_SRC/scripts/generate_world.py" --seed "$SEED" $ARGS
  if [ "$CRIT" = "plaque" ]; then
    PLATE=$(python3 -c "import yaml; print(yaml.safe_load(open('$PKG_SRC/config/parking_slots.yaml'))['gt_target_plate'])")
    QUERY=(color:=aucune "plate:=$PLATE")
  else
    QUERY=(color:=rouge)
  fi
  for ATTEMPT in 1 2; do          # 2e tentative seulement si la mission n'a pas démarré (Gazebo/Nav2 trop lents)
    PIDS=()
    setsid ros2 launch parking_robot parking_world.launch.py gui:=false > "${L}_world.log" 2>&1 & PIDS+=($!)
    sleep 30          # temps de démarrage de Gazebo (valeur réglée par Ahmed)
    setsid ros2 launch parking_robot navigation.launch.py rviz:=false > "${L}_nav.log" 2>&1 & PIDS+=($!)
    sleep 25          # temps de démarrage de Nav2
    setsid ros2 launch parking_robot mission.launch.py "${QUERY[@]}" method:="$METHOD" $EXTRA > "${L}_mission.log" 2>&1 & PIDS+=($!)
    STARTED=0
    for _ in $(seq 1 45); do        # jusqu'à 90 s pour voir « Recherche démarrée »
      if grep -q "Recherche démarrée" "${L}_mission.log" 2>/dev/null; then STARTED=1; break; fi
      sleep 2
    done
    if [ "$STARTED" = 0 ]; then
      echo "Graine $SEED : la mission n'a pas démarré (tentative $ATTEMPT), relance"
      cleanup
      continue
    fi
    timeout "$TIMEOUT" ros2 run parking_robot metrics_logger --ros-args -p use_sim_time:=true \
        -p scenario:="$SC" -p method:="$METHOD-$CRIT" -p seed:="$SEED" -p out:="$HOME/results.csv" \
        || echo "Graine $SEED : pas de résultat (dépassement de temps)"
    cleanup
    break
  done
done
