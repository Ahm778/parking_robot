#!/usr/bin/env python3
"""Banc d'essai de la vision sur la Jetson (ou n'importe quel PC), sans ROS.

Mesure :
  1. la lecture de plaque (Tesseract) sur une image de référence (scripts/data/plaque_test.png) ;
  2. le temps de détection YOLOv5n sur des images de la caméra (TensorRT si le moteur existe, sinon OpenCV) ;
  3. la lecture de plaque : sur les voitures détectées par YOLO, ou directement dans toute l'image
     (--plaque-seule, pour une maquette avec des boîtes ou des voitures jouets que YOLO ne reconnaît pas).

Usage (depuis le dossier du projet) :
    python3 scripts/banc_jetson.py                        # webcam /dev/video0, 30 images
    python3 scripts/banc_jetson.py --moteur opencv        # forcer OpenCV (comparaison)
    python3 scripts/banc_jetson.py --plaque-seule         # chercher la plaque dans toute l'image, sans YOLO
    python3 scripts/banc_jetson.py --camera 1 --images 50
    python3 scripts/banc_jetson.py --sans-camera          # test de plaque + YOLO sur image fixe

Moteur TensorRT (une fois, sur la Jetson, 5 à 15 minutes) :
    /usr/src/tensorrt/bin/trtexec --onnx=weights/yolov5n.onnx --saveEngine=weights/yolov5n_fp16.engine --fp16

Écrit /tmp/banc_vision.jpg : la dernière image avec les détections dessinées.
Compatible Python 3.6 (Jetson Nano, JetPack 4.6).
"""
import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from parking_robot.plates import find_plate_scene, read_plate, read_plate_robust   # noqa: E402
from parking_robot.yolo_fast import COCO, charger_yolo                           # noqa: E402


def ms(values):
    return 'moyenne %.0f ms, médiane %.0f ms, max %.0f ms' % (
        statistics.mean(values), statistics.median(values), max(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camera', type=int, default=0, help='numéro de la caméra (/dev/videoN), 0 par défaut')
    ap.add_argument('--images', type=int, default=30, help="nombre d'images à mesurer")
    ap.add_argument('--taille', type=int, default=320, help="taille d'entrée de YOLO pour OpenCV (320)")
    ap.add_argument('--moteur', choices=['auto', 'tensorrt', 'opencv'], default='auto')
    ap.add_argument('--plaque-seule', action='store_true',
                    help='chercher la plaque dans toute l\'image, sans YOLO (maquette)')
    ap.add_argument('--sans-camera', action='store_true', help='ne pas ouvrir la caméra')
    a = ap.parse_args()

    print('OpenCV', cv2.__version__, '| Python', sys.version.split()[0])

    # 1. Lecture de plaque sur l'image de référence
    ref = cv2.imread(os.path.join(ROOT, 'scripts', 'data', 'plaque_test.png'))
    t, lu = [], ''
    for _ in range(5):
        t0 = time.time()
        lu = read_plate(ref)
        t.append((time.time() - t0) * 1000)
    print('\n[1] Plaque de référence 187-3985 -> lu : %r  (%s)' % (lu, 'OK' if lu == '187-3985' else 'ÉCHEC'))
    print('    temps Tesseract :', ms(t))

    yolo = None
    if not a.plaque_seule:
        yolo = charger_yolo(os.path.join(ROOT, 'weights'), a.moteur, a.taille)
        print('\n    détecteur :', yolo.nom)
        yolo(np.zeros((480, 640, 3), np.uint8))                   # premier appel (initialisation), non compté

    if a.sans_camera:
        if yolo is not None:
            img = np.full((480, 640, 3), 127, np.uint8)
            t = []
            for _ in range(a.images):
                t0 = time.time()
                yolo(img)
                t.append((time.time() - t0) * 1000)
            print('\n[2] YOLO sur image fixe :', ms(t), '-> %.1f images/s' % (1000 / statistics.mean(t)))
        return

    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, img = cap.read()
    if not ok:
        sys.exit('Caméra %d illisible : vérifiez « ls /dev/video* » et le numéro passé à --camera.' % a.camera)
    print('    caméra %d : %dx%d' % (a.camera, img.shape[1], img.shape[0]))

    t_cap, t_det, t_ocr, vus, plaques = [], [], [], {}, []
    for _ in range(a.images):
        t0 = time.time()
        ok, img = cap.read()
        t1 = time.time()
        if not ok:
            continue
        t_cap.append((t1 - t0) * 1000)
        if a.plaque_seule:                                         # maquette : plaque cherchée dans toute l'image
            roi = find_plate_scene(img)
            t2 = time.time()
            t_det.append((t2 - t1) * 1000)
            if roi is not None:
                vus['plaque'] = vus.get('plaque', 0) + 1
                p = read_plate(roi)
                t_ocr.append((time.time() - t2) * 1000)
                if p:
                    plaques.append(p)
            continue
        dets = yolo(img)
        t_det.append((time.time() - t1) * 1000)
        for x1, y1, x2, y2, conf, cls in dets:
            vus[COCO[cls]] = vus.get(COCO[cls], 0) + 1
            if cls in (2, 5, 7):                                   # plaque sur le véhicule détecté
                X1, Y1 = max(0, int(x1)), max(0, int(y1))
                X2, Y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
                if X2 - X1 > 20 and Y2 - Y1 > 20:
                    t3 = time.time()
                    p = read_plate_robust(img[Y1:Y2, X1:X2])
                    t_ocr.append((time.time() - t3) * 1000)
                    if p:
                        plaques.append(p)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, '%s %.2f' % (COCO[cls], conf), (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cap.release()
    cv2.imwrite('/tmp/banc_vision.jpg', img)

    quoi = 'recherche de plaque' if a.plaque_seule else 'YOLO (%s)' % yolo.nom
    print('\n[2] Sur %d images de la caméra :' % len(t_det))
    print('    lecture caméra :', ms(t_cap))
    print('    %s :' % quoi, ms(t_det), '-> %.1f images/s' % (1000 / statistics.mean(t_det)))
    print('    objets vus (total sur toutes les images) :', vus or 'aucun')
    print('\n[3] Lecture de plaque :')
    if t_ocr:
        print('    %d tentatives,' % len(t_ocr), ms(t_ocr))
        print('    plaques lues :', sorted(set(plaques)) or 'aucune')
    else:
        print('    aucune plaque ni voiture trouvée (normal si rien devant la caméra)')
    print('\nImage annotée : /tmp/banc_vision.jpg')


if __name__ == '__main__':
    main()
