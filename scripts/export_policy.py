#!/usr/bin/env python3
"""Exporte la politique PPO (Stable-Baselines3) en weights/policy.npz, lisible avec NumPy seul.

Usage : python3 scripts/export_policy.py ppo_parking_level2.zip
Le nœud drl_controller (et la Jetson) n'ont alors besoin ni de PyTorch ni de Stable-Baselines3.
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking_robot.drl_common import NumpyPolicy  # noqa: E402
from parking_robot.drl_env import ParkingNavEnv  # noqa: E402


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else 'ppo_parking_level2.zip'
    model = PPO.load(zip_path, device='cpu')
    sd = {k: v.cpu().numpy() for k, v in model.policy.state_dict().items()}
    out = Path(__file__).resolve().parents[1] / 'weights' / 'policy.npz'
    out.parent.mkdir(exist_ok=True)
    np.savez(out,
             w0=sd['mlp_extractor.policy_net.0.weight'], b0=sd['mlp_extractor.policy_net.0.bias'],
             w1=sd['mlp_extractor.policy_net.2.weight'], b1=sd['mlp_extractor.policy_net.2.bias'],
             w2=sd['action_net.weight'], b2=sd['action_net.bias'])
    # Vérification : la politique NumPy choisit les mêmes actions que le modèle PyTorch
    pol, env = NumpyPolicy(out), ParkingNavEnv(level=2)
    same = 0
    for i in range(200):
        obs, _ = env.reset(seed=i)
        same += int(pol.act(obs) == int(model.predict(obs, deterministic=True)[0]))
    print(f'{out} écrit ; actions identiques sur {same}/200 observations')


if __name__ == '__main__':
    main()
