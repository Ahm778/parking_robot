#!/usr/bin/env python3
"""Plaques de la maquette : feuille A4 à imprimer + modèles de chiffres pour la lecture fiable.

À lancer sur un PC (Pillow récent), pas sur la Jetson :
    python3 scripts/imprimer_plaques.py                          # 8 plaques au hasard -> plaques_a_imprimer.pdf
    python3 scripts/imprimer_plaques.py --plaques 187-3985 52-1204 213-77
    python3 scripts/imprimer_plaques.py --largeur-cm 12 --nombre 6 --graine 4
    python3 scripts/imprimer_plaques.py --modeles                # refait scripts/data/chiffres_plaque.npz

Imprimer à 100 % (« taille réelle », pas « ajuster à la page »). Les numéros sont inventés.
Les modèles de chiffres sont fabriqués avec exactement la même police et le même traitement que la lecture
(parking_robot/plaque_fiable.py) : c'est ce qui rend la lecture fiable.
"""
import argparse
import os
import random
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from parking_robot.plates import display, random_plate, render_plate_png   # noqa: E402
from parking_robot import plaque_fiable as pf                                # noqa: E402

DPI = 300


def image_plaque(plaque, px_w=1400):
    chemin = os.path.join(tempfile.gettempdir(), 'plaque_%s.png' % plaque)
    render_plate_png(plaque, chemin, px_w=px_w, px_h=int(px_w * 150 / 700))
    return cv2.imread(chemin)


def scene_aleatoire(plaque_img, rng):
    """La plaque posée dans une scène, vue de biais, floutée, bruitée, plus ou moins éclairée."""
    h0, w0 = 480, 640
    fond = np.full((h0, w0, 3), rng.randint(60, 230), np.uint8)
    fond = cv2.add(fond, rng.normal(0, 12, fond.shape).astype(np.int16).clip(-40, 40).astype(np.uint8))
    larg = rng.uniform(110, 420)
    haut = larg * plaque_img.shape[0] / plaque_img.shape[1]
    cx, cy = rng.uniform(larg / 2 + 10, w0 - larg / 2 - 10), rng.uniform(haut + 10, h0 - haut - 10)
    lacet = np.radians(rng.uniform(-45, 45))              # vue de côté : la plaque paraît plus étroite
    roulis = np.radians(rng.uniform(-12, 12))
    lx = larg / 2 * np.cos(lacet)
    pers = rng.uniform(0, 0.18) * np.sign(lacet) * haut
    coins = np.float32([[-lx, -haut / 2 - pers], [lx, -haut / 2 + pers], [lx, haut / 2 - pers], [-lx, haut / 2 + pers]])
    R = np.float32([[np.cos(roulis), -np.sin(roulis)], [np.sin(roulis), np.cos(roulis)]])
    coins = coins.dot(R.T) + np.float32([cx, cy])
    h, w = plaque_img.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), coins)
    warp = cv2.warpPerspective(plaque_img, M, (w0, h0))
    masque = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (w0, h0))
    fond[masque > 0] = warp[masque > 0]
    gain = rng.uniform(0.55, 1.25)
    img = (fond.astype(np.float32) * gain + rng.uniform(-25, 25)).clip(0, 255).astype(np.uint8)
    k = rng.choice([1, 3, 3, 5])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = (img.astype(np.int16) + rng.normal(0, rng.uniform(2, 9), img.shape)).clip(0, 255).astype(np.uint8)
    return img


def fabriquer_modeles(n_plaques=60, graine=0):
    rng = np.random.RandomState(graine)
    prng = random.Random(graine)
    images, labels = [], []
    for _ in range(n_plaques):
        p = random_plate(prng)
        texte = p.replace('-', '')
        img_p = image_plaque(p, px_w=700)
        for _ in range(4):
            scene = scene_aleatoire(img_p, rng)
            for q in pf.trouver_plaques(scene):
                g = pf.chiffres(pf.redresser(scene, q))
                if g is None or len(g[0]) + len(g[1]) != len(texte):
                    continue
                for c, lab in zip(g[0] + g[1], texte):
                    images.append(c.astype(np.uint8))
                    labels.append(int(lab))
                break
    os.makedirs(os.path.dirname(pf.MODELES), exist_ok=True)
    np.savez_compressed(pf.MODELES, images=np.array(images), labels=np.array(labels))
    print('modèles :', len(labels), 'chiffres ->', pf.MODELES)


def feuille(plaques, largeur_cm, sortie):
    from PIL import Image, ImageDraw
    a4 = (int(21.0 / 2.54 * DPI), int(29.7 / 2.54 * DPI))
    page = Image.new('RGB', a4, 'white')
    d = ImageDraw.Draw(page)
    pw = int(largeur_cm / 2.54 * DPI)
    ph = int(pw * 150 / 700)
    marge, esp = int(1.2 / 2.54 * DPI), int(0.9 / 2.54 * DPI)
    x, y = marge, marge
    for p in plaques:
        if x + pw > a4[0] - marge:
            x, y = marge, y + ph + esp
        if y + ph > a4[1] - marge:
            break
        img = Image.fromarray(cv2.cvtColor(image_plaque(p, px_w=pw), cv2.COLOR_BGR2RGB))
        page.paste(img, (x, y))
        d.rectangle([x - 12, y - 12, x + pw + 12, y + ph + 12], outline=(170, 170, 170), width=2)   # trait de coupe
        d.text((x, y + ph + 16), display(p), fill=(90, 90, 90))
        x += pw + esp
    page.save(sortie, 'PDF', resolution=DPI)
    page.save(os.path.splitext(sortie)[0] + '.png')
    print('feuille :', sortie, '(imprimer à 100 %%, plaques de %.1f cm)' % largeur_cm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plaques', nargs='*', help='plaques « série-numéro », ex. 187-3985')
    ap.add_argument('--nombre', type=int, default=8)
    ap.add_argument('--graine', type=int, default=1)
    ap.add_argument('--largeur-cm', type=float, default=12.0)
    ap.add_argument('--sortie', default='plaques_a_imprimer.pdf')
    ap.add_argument('--modeles', action='store_true', help='refaire les modèles de chiffres')
    a = ap.parse_args()
    if a.modeles or not os.path.exists(pf.MODELES):
        fabriquer_modeles()
    plaques = a.plaques or [random_plate(random.Random(a.graine * 1000 + i)) for i in range(a.nombre)]
    feuille(plaques, a.largeur_cm, a.sortie)


if __name__ == '__main__':
    main()
