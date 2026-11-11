#!/usr/bin/env python3
"""Entraîne l'agent PPO dans l'environnement 2D avec un curriculum (niveaux 0 -> 1 -> 2).

Usage : python3 scripts/train_ppo.py [--steps 200000 600000 1200000] [--algo ppo|dqn]
Produit runs/ (TensorBoard) et ppo_parking_level{0,1,2}.zip, puis évalue chaque niveau.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking_robot.drl_env import ParkingNavEnv  # noqa: E402


def evaluate(model, level, episodes=100, seed=1000, ped_yield=True):
    env = ParkingNavEnv(level=level, ped_yield=ped_yield)
    succ = coll = stopped = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        succ += info['success']
        coll += info['collision']
        stopped += info['hit_while_stopped']
    return succ / episodes, coll / episodes, stopped / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, nargs=3, default=[200_000, 600_000, 1_200_000])
    ap.add_argument('--algo', choices=['ppo', 'dqn'], default='ppo')
    ap.add_argument('--noise', type=float, default=0.02, help='bruit gaussien sur le lidar (aléa de domaine)')
    a = ap.parse_args()
    model = None
    for level, steps in enumerate(a.steps):
        if a.algo == 'ppo':
            env = make_vec_env(lambda: ParkingNavEnv(level=level, noise=a.noise), n_envs=8, seed=42 + level)
            if model is None:
                model = PPO('MlpPolicy', env, seed=42, verbose=1, tensorboard_log='runs', device='cpu',
                            ent_coef=0.01)   # garde de l'exploration : évite que la politique se fige
            model.set_env(env)
        else:
            env = ParkingNavEnv(level=level, noise=a.noise)
            if model is None:
                model = DQN('MlpPolicy', env, seed=42, verbose=1, tensorboard_log='runs', device='cpu',
                            buffer_size=100_000, learning_starts=5_000, exploration_fraction=0.2)
            model.set_env(env)
        model.learn(steps, reset_num_timesteps=False, tb_log_name=f'{a.algo}_parking')
        model.save(f'{a.algo}_parking_level{level}')
        s, c, cs = evaluate(model, level)
        print(f'== Niveau {level} : réussite {s:.0%}, collisions {c:.0%} '
              f'(dont {cs:.0%} robot arrêté, percuté par un piéton) (100 épisodes)')
        if level == 2:
            s, c, cs = evaluate(model, level, ped_yield=False)
            print(f'   test de stress, piétons qui ne s\'arrêtent jamais : réussite {s:.0%}, collisions {c:.0%} '
                  f'(dont {cs:.0%} robot arrêté)')
    print('Terminé. Exportez la politique avec scripts/export_policy.py')


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    main()
