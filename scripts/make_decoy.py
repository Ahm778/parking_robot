#!/usr/bin/env python3
"""Crée le leurre « suv_red » (scénario S6) : copie du modèle suv avec une texture teintée en rouge.

Usage : python3 scripts/make_decoy.py ~/gazebo_models
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '~/gazebo_models').expanduser()
    src, dst = root / 'suv', root / 'suv_red'
    if not src.is_dir():
        sys.exit(f'Modèle introuvable : {src}')
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for name in ('model.sdf', 'model.config'):
        f = dst / name
        f.write_text(f.read_text().replace('model://suv/', 'model://suv_red/')
                     .replace('<name>SUV</name>', '<name>SUV red</name>'))
    tex = dst / 'materials' / 'textures' / 'suv.png'
    img = np.asarray(Image.open(tex).convert('RGB')).astype(float)
    grey = img.mean(axis=2, keepdims=True)
    red = np.clip(grey * np.array([1.6, 0.35, 0.35]), 0, 255).astype(np.uint8)   # carrosserie rouge
    Image.fromarray(red).save(tex)
    print(f'Leurre créé : {dst}')


if __name__ == '__main__':
    main()
