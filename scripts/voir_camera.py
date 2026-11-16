#!/usr/bin/env python3
"""Affiche en direct ce que voit la caméra, avec les détections YOLO et la lecture de plaque.

Usage (depuis le dossier du projet, sur la Jetson avec un écran, ou sur un PC) :
    python3 scripts/voir_camera.py                 # webcam /dev/video0, YOLO (TensorRT si disponible)
    python3 scripts/voir_camera.py --plaque-seule  # lecture FIABLE des plaques imprimées (sans YOLO, en continu)
    python3 scripts/voir_camera.py --brut          # image seule, sans aucun traitement
    python3 scripts/voir_camera.py --camera 1

Touches (fenêtre active) :
    l : lire la plaque maintenant        s : enregistrer l'image dans /tmp
    q ou Échap : quitter

Sur la Jetson, OpenCV installé par pip est « headless » (sans fenêtre) : le script se relance alors tout
seul avec l'OpenCV de JetPack (python3 -s), qui sait ouvrir une fenêtre. Dans ce cas YOLO passe par
TensorRT (weights/yolov5n_fp16.engine, voir yolo_fast.py) : l'OpenCV de JetPack ne lit pas le modèle ONNX.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def fenetre_possible():
    try:
        cv2.namedWindow('test')
        cv2.destroyWindow('test')
        return True
    except cv2.error:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--moteur', choices=['auto', 'tensorrt', 'opencv'], default='auto')
    ap.add_argument('--plaque-seule', action='store_true', help='sans YOLO : chercher la plaque dans toute l\'image')
    ap.add_argument('--brut', action='store_true', help='image seule, sans traitement')
    a = ap.parse_args()

    if not fenetre_possible():
        if not sys.flags.no_user_site:                       # relance avec l'OpenCV système (avec fenêtres)
            print('OpenCV %s sans fenêtre : relance avec l\'OpenCV de JetPack...' % cv2.__version__)
            os.execv(sys.executable, [sys.executable, '-s'] + sys.argv)
        sys.exit('Aucun OpenCV avec fenêtre disponible. Utilisez scripts/banc_jetson.py (image dans /tmp).')

    from parking_robot.plates import find_plate_scene, read_plate, read_plate_robust
    from parking_robot.yolo_fast import COCO, charger_yolo
    lecteur = None
    if a.plaque_seule:
        from parking_robot.plaque_fiable import LecteurFiable
        lecteur = LecteurFiable()

    yolo = None
    if not (a.brut or a.plaque_seule):
        yolo = charger_yolo(os.path.join(ROOT, 'weights'), a.moteur)
        print('OpenCV', cv2.__version__, '| détecteur :', yolo.nom)

    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit('Caméra %d introuvable : vérifiez « ls /dev/video* ».' % a.camera)

    titre = 'Camera du robot (q pour quitter, l pour lire la plaque)'
    cv2.namedWindow(titre)
    derniere_plaque, t_prec, fps = '', time.time(), 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            print('Image non reçue, nouvel essai...')
            time.sleep(0.2)
            continue
        vue = img.copy()
        voitures, plaque_roi = [], None

        if yolo is not None:
            for x1, y1, x2, y2, conf, cls in yolo(img):
                couleur = (0, 200, 0) if cls != 0 else (0, 140, 255)
                cv2.rectangle(vue, (int(x1), int(y1)), (int(x2), int(y2)), couleur, 2)
                cv2.putText(vue, '%s %.2f' % (COCO[cls], conf), (int(x1), max(15, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 2)
                if cls in (2, 5, 7):
                    voitures.append((int(x1), int(y1), int(x2), int(y2)))
        elif lecteur is not None:                           # lecture fiable, à chaque image + vote
            confirme, lu, q = lecteur.mettre_a_jour(img)
            if q is not None:
                couleur = (0, 220, 0) if confirme else ((0, 220, 255) if lu else (0, 0, 255))
                cv2.polylines(vue, [q.astype(np.int32).reshape(-1, 1, 2)], True, couleur, 2)
            cv2.putText(vue, 'image : %s' % (lu or '-'), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            if confirme and confirme != derniere_plaque:
                print('PLAQUE CONFIRMEE : %s' % confirme)
            derniere_plaque = confirme

        maintenant = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-3, maintenant - t_prec)
        t_prec = maintenant
        cv2.putText(vue, '%.1f images/s' % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if derniere_plaque:
            cv2.putText(vue, ('CONFIRMEE : %s' if lecteur is not None else 'plaque : %s') % derniere_plaque,
                        (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(titre, vue)

        touche = cv2.waitKey(1) & 0xFF
        if touche in (ord('q'), 27):
            break
        if touche == ord('s'):
            chemin = '/tmp/camera_%d.jpg' % int(time.time())
            cv2.imwrite(chemin, img)
            print('image enregistrée :', chemin)
        if touche == ord('l') and lecteur is None:          # lecture à la demande (≈ 0,4 s sur la Jetson)
            t0 = time.time()
            lu = ''
            if plaque_roi is not None:
                lu = read_plate(plaque_roi)
            elif voitures:
                x1, y1, x2, y2 = max(voitures, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                lu = read_plate_robust(img[max(0, y1):y2, max(0, x1):x2])
            else:
                roi = find_plate_scene(img)
                lu = read_plate(roi) if roi is not None else ''
            derniere_plaque = lu or '(rien lu)'
            print('plaque : %s  (%.0f ms)' % (derniere_plaque, (time.time() - t0) * 1000))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
