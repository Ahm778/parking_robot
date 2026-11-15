#!/usr/bin/env python3
"""Génère la carte Nav2 du parking (maps/parking.pgm + maps/parking.yaml) : murs et butées, sans les voitures.

Les voitures changent à chaque essai : elles ne sont pas sur la carte, le costmap les découvre au lidar.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math

from parking_robot.geometry import WALL_X, WALL_Y, map_obstacles  # noqa: E402

RES = 0.05
SIZE = 2 * math.ceil(max(WALL_X, WALL_Y) + 0.5)    # 20 m en mode normal, 22 m en mode démo                      # m par pixel, côté de la carte (m), centrée sur l'origine
N = int(round(SIZE / RES))


def fill(img, b):
    xmin, ymin, xmax, ymax = b
    c0, c1 = int(np.floor((xmin + SIZE / 2) / RES)), int(np.ceil((xmax + SIZE / 2) / RES))
    r0, r1 = int(np.floor((SIZE / 2 - ymax) / RES)), int(np.ceil((SIZE / 2 - ymin) / RES))
    img[max(r0, 0):min(r1, N), max(c0, 0):min(c1, N)] = 0


def main():
    out = Path(__file__).resolve().parents[1] / 'maps'
    out.mkdir(exist_ok=True)
    img = np.full((N, N), 254, np.uint8)       # 254 = libre, 0 = occupé (ligne 0 de l'image = y maximal)
    for b in map_obstacles():
        fill(img, b)
    Image.fromarray(img).save(out / 'parking.pgm')
    with open(out / 'parking.yaml', 'w') as f:
        yaml.safe_dump({'image': 'parking.pgm', 'mode': 'trinary', 'resolution': RES,
                        'origin': [-SIZE / 2, -SIZE / 2, 0.0], 'negate': 0,
                        'occupied_thresh': 0.65, 'free_thresh': 0.25}, f, sort_keys=False)
    print(f'Carte écrite dans {out} ({N} x {N} pixels, {RES} m/pixel)')


if __name__ == '__main__':
    main()
