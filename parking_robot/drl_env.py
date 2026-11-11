"""Environnement Gymnasium 2D du parking pour entraîner l'agent DRL (rapide, sans Gazebo).

Même géométrie que le monde Gazebo (geometry.py), lidar simplifié à 24 rayons.
Curriculum : level 0 = allée vide, 1 = voitures + obstacles fixes, 2 = + piétons mobiles.
Piétons : par défaut ils s'arrêtent quand le robot est à moins de YIELD_DIST d'eux (comme un vrai piéton,
et comme pedestrian_mover dans Gazebo). ped_yield=False donne des piétons « aveugles » (test de stress).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from parking_robot.drl_common import ACTIONS, DT, MAX_R, N_RAYS, OBS_SIZE, REL_ANGLES, ROBOT_R, build_obs
from parking_robot.geometry import AISLE_HALF, ROW_SIGN, XS, car_box, stops, walls


def ray_box(ox, oy, dx, dy, b):
    """Distance le long du rayon jusqu'au rectangle b = (xmin, ymin, xmax, ymax), inf si pas d'intersection."""
    tmin, tmax = 0.0, np.inf
    for o, d, lo, hi in ((ox, dx, b[0], b[2]), (oy, dy, b[1], b[3])):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return np.inf
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
            if tmin > tmax:
                return np.inf
    return tmin


YIELD_DIST = 0.8      # distance (m) à laquelle un piéton s'arrête pour laisser passer le robot


class ParkingNavEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, level=0, max_steps=400, noise=0.0, ped_yield=True):
        self.level, self.max_steps, self.noise, self.ped_yield = level, max_steps, noise, ped_yield
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), np.float32)
        self.fixed = walls() + stops()

    def boxes(self):
        peds = [(x - .25, y - .25, x + .25, y + .25) for x, y, _ in self.peds]
        return self.fixed + self.static + peds

    def free(self, x, y, margin=0.6):
        return all(not (b[0] - margin < x < b[2] + margin and b[1] - margin < y < b[3] + margin)
                   for b in self.boxes())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.static, self.peds = [], []
        if self.level >= 1:
            self.static = [car_box(x, row) for row in ROW_SIGN for x in XS if r.random() < 0.7]
            for _ in range(int(r.integers(1, 4))):                   # obstacles fixes dans l'allée
                x, y = r.uniform(-6, 6), r.uniform(-2, 2)
                self.static.append((x - .3, y - .3, x + .3, y + .3))
        if self.level >= 2:                                          # piétons en va-et-vient
            self.peds = [[r.uniform(-6, 6), r.uniform(-2.5, 2.5), r.choice([-1, 1]) * r.uniform(.3, .7)]
                         for _ in range(int(r.integers(1, 3)))]
        lim = AISLE_HALF - 0.5
        while True:
            self.pose = np.array([r.uniform(-8, 7), r.uniform(-lim, lim), r.uniform(-np.pi, np.pi)])
            self.goal = np.array([r.uniform(-7.5, 7.5), r.uniform(-lim, lim)])
            if self.free(*self.pose[:2]) and self.free(*self.goal) and \
               np.hypot(*(self.goal - self.pose[:2])) > 1.5:
                break
        self.steps, self.v, self.w, self.prev_a = 0, 0.0, 0.0, 4
        self.prev_d = float(np.hypot(*(self.goal - self.pose[:2])))
        self.prev_rays = None
        return self.obs(), {}

    def obs(self):
        x, y, th = self.pose
        boxes = self.boxes()
        self.rays = np.array([min([MAX_R] + [ray_box(x, y, np.cos(th + a), np.sin(th + a), b) for b in boxes])
                              for a in REL_ANGLES])
        rays = np.clip(self.rays + (self.np_random.normal(0, self.noise, N_RAYS) if self.noise else 0.0), 0, MAX_R)
        prev = rays if self.prev_rays is None else self.prev_rays
        self.prev_rays = rays
        dx, dy = self.goal - self.pose[:2]
        ang = np.arctan2(dy, dx) - th
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        return build_obs(rays, prev, np.hypot(dx, dy), ang, self.v, self.w)

    def step(self, a):
        a = int(a)
        self.v, self.w = ACTIONS[a]
        x, y, th = self.pose
        th += self.w * DT
        self.pose = np.array([x + self.v * np.cos(th) * DT, y + self.v * np.sin(th) * DT,
                              np.arctan2(np.sin(th), np.cos(th))])
        for p in self.peds:
            if self.ped_yield and np.hypot(p[0] - self.pose[0], p[1] - self.pose[1]) < YIELD_DIST:
                continue                                  # le piéton laisse passer le robot
            p[1] += p[2] * DT
            if abs(p[1]) > 2.5:
                p[2] *= -1
        self.steps += 1
        o = self.obs()
        d = float(np.hypot(*(self.goal - self.pose[:2])))
        rew = 2.0 * (self.prev_d - d) - 0.01 - 0.05 * (a != self.prev_a)      # progrès, temps, à-coups
        rew -= 0.2 * max(0.0, 0.5 - self.rays.min())                          # trop près d'un obstacle
        for px, py, _ in self.peds:                                           # trop près d'un piéton
            rew -= 0.5 * max(0.0, 1.0 - np.hypot(px - self.pose[0], py - self.pose[1]))
        self.prev_d, self.prev_a = d, a
        collision, success = bool(self.rays.min() < ROBOT_R), d < 0.3
        rew += -10.0 if collision else (10.0 if success else 0.0)
        return o, float(rew), collision or success, self.steps >= self.max_steps, \
            {'collision': collision, 'success': success, 'hit_while_stopped': collision and self.v == 0.0}
