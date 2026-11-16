#!/usr/bin/env python3
"""Banc d'essai de la vision sur la Jetson (ou n'importe quel PC), sans ROS.

Mesure :
  1. la lecture de plaque (Tesseract) sur une image de référence (scripts/data/plaque_test.png) ;
  2. le temps de détection YOLOv5n (OpenCV DNN) sur des images de la caméra ;
  3. en option, la lecture de plaque sur les voitures détectées par la caméra.

Usage (depuis le dossier du projet) :
    python3 scripts/banc_jetson.py                 # webcam /dev/video0, 30 images
    python3 scripts/banc_jetson.py --camera 1 --images 50
    python3 scripts/banc_jetson.py --sans-camera   # seulement le test de plaque et un test YOLO sur image fixe

Écrit /tmp/banc_vision.jpg : la dernière image avec les détections dessinées.
Compatible Python 3.6 (Jetson Nano, JetPack 4.6). Nécessite OpenCV >= 4.5 pour lire le modèle YOLO.
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
from parking_robot.plates import read_plate, read_plate_robust   # noqa: E402  (pas de ROS dans ce module)

COCO = {0: 'pieton', 2: 'voiture', 5: 'bus', 7: 'utilitaire'}


class YoloOnnx:
    """Même code que dans vision_node.py (copié pour ne pas dépendre de ROS)."""

    def __init__(self, path, size=320, conf=0.25, nms=0.45):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.size, self.conf, self.nms = size, conf, nms

    def __call__(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0]
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = (conf > self.conf) & np.isin(cls, list(COCO))
        out, cls, conf = out[keep], cls[keep], conf[keep]
        sx, sy = w / self.size, h / self.size
        boxes = [[float((cx - bw / 2) * sx), float((cy - bh / 2) * sy), float(bw * sx), float(bh * sy)]
                 for cx, cy, bw, bh in out[:, :4]]
        if not boxes:
            return []
        idx = np.array(cv2.dnn.NMSBoxes(boxes, conf.tolist(), self.conf, self.nms)).flatten()
        return [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3],
                 float(conf[i]), int(cls[i])) for i in idx]


def ms(values):
    return 'moyenne %.0f ms, médiane %.0f ms, max %.0f ms' % (
        statistics.mean(values), statistics.median(values), max(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camera', type=int, default=0, help='numéro de la caméra (/dev/videoN), 0 par défaut')
    ap.add_argument('--images', type=int, default=30, help="nombre d'images à mesurer")
    ap.add_argument('--taille', type=int, default=320, help='taille d\'entrée de YOLO (320 comme sur le robot)')
    ap.add_argument('--sans-camera', action='store_true', help='ne pas ouvrir la caméra')
    a = ap.parse_args()

    print('OpenCV', cv2.__version__, '| Python', sys.version.split()[0])

    # 1. Lecture de plaque sur l'image de référence
    ref = cv2.imread(os.path.join(ROOT, 'scripts', 'data', 'plaque_test.png'))
    t = []
    lu = ''
    for _ in range(5):
        t0 = time.time()
        lu = read_plate(ref)
        t.append((time.time() - t0) * 1000)
    print('\n[1] Plaque de référence 187-3985 -> lu : %r  (%s)' % (lu, 'OK' if lu == '187-3985' else 'ÉCHEC'))
    print('    temps Tesseract :', ms(t))

    # 2. YOLO
    yolo = YoloOnnx(os.path.join(ROOT, 'weights', 'yolov5n.onnx'), size=a.taille)
    yolo(np.zeros((480, 640, 3), np.uint8))                       # premier appel (initialisation), non compté
    if a.sans_camera:
        img = np.full((480, 640, 3), 127, np.uint8)
        t = []
        for _ in range(a.images):
            t0 = time.time()
            yolo(img)
            t.append((time.time() - t0) * 1000)
        print('\n[2] YOLO %dx%d sur image fixe :' % (a.taille, a.taille), ms(t),
              '-> %.1f images/s' % (1000 / statistics.mean(t)))
        return

    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, img = cap.read()
    if not ok:
        sys.exit('Caméra %d illisible : vérifiez « ls /dev/video* » et le numéro passé à --camera.' % a.camera)
    print('\n    caméra %d : %dx%d' % (a.camera, img.shape[1], img.shape[0]))

    t_cap, t_yolo, t_ocr, vus, plaques = [], [], [], {}, []
    for _ in range(a.images):
        t0 = time.time()
        ok, img = cap.read()
        t1 = time.time()
        if not ok:
            continue
        dets = yolo(img)
        t2 = time.time()
        t_cap.append((t1 - t0) * 1000)
        t_yolo.append((t2 - t1) * 1000)
        for x1, y1, x2, y2, conf, cls in dets:
            vus[COCO[cls]] = vus.get(COCO[cls], 0) + 1
            if cls in (2, 5, 7):                                   # 3. plaque sur la voiture détectée
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
                if x2 - x1 > 20 and y2 - y1 > 20:
                    t3 = time.time()
                    p = read_plate_robust(img[y1:y2, x1:x2])
                    t_ocr.append((time.time() - t3) * 1000)
                    if p:
                        plaques.append(p)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, '%s %.2f' % (COCO[cls], conf), (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cap.release()
    cv2.imwrite('/tmp/banc_vision.jpg', img)

    print('\n[2] Sur %d images de la caméra :' % len(t_yolo))
    print('    lecture caméra :', ms(t_cap))
    print('    YOLO %dx%d     :' % (a.taille, a.taille), ms(t_yolo), '-> %.1f images/s' % (1000 / statistics.mean(t_yolo)))
    print('    objets vus (total sur toutes les images) :', vus or 'aucun')
    print('\n[3] Lecture de plaque sur les voitures détectées :')
    if t_ocr:
        print('    %d tentatives,' % len(t_ocr), ms(t_ocr))
        print('    plaques lues :', sorted(set(plaques)) or 'aucune')
    else:
        print('    aucune voiture détectée (normal sans voiture devant la caméra)')
    print('\nImage annotée : /tmp/banc_vision.jpg')


if __name__ == '__main__':
    main()
