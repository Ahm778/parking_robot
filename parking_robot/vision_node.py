"""Détection (YOLOv5n ONNX via OpenCV DNN) + identification (couleur, type, plaque) + confirmation multi-images.

Requête : target_color, target_type, target_plate ; « aucune » (ou vide) = critère ignoré.
Si target_plate est donnée, la plaque lue doit lui ressembler (similarité >= plate_threshold) pour qu'un
véhicule puisse être confirmé : c'est elle qui départage deux voitures de même couleur.

Publie :
  /vision/target_candidate  (std_msgs/String, JSON) : meilleur véhicule candidat
  /vision/pedestrians       (std_msgs/String, JSON) : liste des piétons vus
  /vision/observations      (std_msgs/String, JSON) : toutes les voitures vues (couleur, plaque lue)
  /vision/debug             (sensor_msgs/Image)     : image annotée
"""
import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from parking_robot.common import share_path
from parking_robot.plates import plate_similarity, read_plate_robust

HUES = {'rouge': [(0, 10), (170, 179)], 'orange': [(11, 22)], 'jaune': [(23, 34)],
        'vert': [(35, 85)], 'bleu': [(86, 130)], 'violet': [(131, 169)]}     # teinte OpenCV : 0-179
COCO = {0: 'pieton', 2: 'voiture', 5: 'bus', 7: 'utilitaire'}             # classes COCO utiles


def color_score(crop, color):
    """Part des pixels de la couleur demandée, ramenée entre 0 et 1."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if color == 'blanc':
        m = (s < 40) & (v > 180)
    elif color == 'noir':
        m = v < 50
    elif color == 'gris':
        m = (s < 40) & (v >= 50) & (v <= 180)
    else:
        m = np.zeros(h.shape, bool)
        for lo, hi in HUES[color]:
            m |= (h >= lo) & (h <= hi)
        m &= (s > 80) & (v > 50)
    return min(1.0, float(m.mean()) / 0.35)      # 35 % de pixels de la bonne couleur = score maximal


class YoloOnnx:
    """YOLOv5n exporté en ONNX (sortie 1 x N x 85), exécuté par OpenCV DNN : pas besoin de PyTorch."""

    def __init__(self, path, size=320, conf=0.2, nms=0.45):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.size, self.conf, self.nms = size, conf, nms

    def __call__(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0]                       # N x 85 : cx, cy, bw, bh, objectness, 80 classes
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = (conf > self.conf) & np.isin(cls, list(COCO))
        out, cls, conf = out[keep], cls[keep], conf[keep]
        sx, sy = w / self.size, h / self.size
        boxes = [[float((cx - bw / 2) * sx), float((cy - bh / 2) * sy), float(bw * sx), float(bh * sy)]
                 for cx, cy, bw, bh in out[:, :4]]
        idx = np.array(cv2.dnn.NMSBoxes(boxes, conf.tolist(), self.conf, self.nms)).flatten()
        return [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3],
                 float(conf[i]), int(cls[i])) for i in idx]


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.declare_parameter('target_color', 'rouge')
        self.declare_parameter('target_type', 'voiture')
        self.declare_parameter('score_threshold', 0.6)
        self.declare_parameter('confirm_frames', 5)
        self.declare_parameter('model_path', share_path('weights', 'yolov5n.onnx'))
        self.declare_parameter('input_size', 320)
        self.declare_parameter('yolo_conf', 0.2)          # 0,2 : les voitures Gazebo sont détectées avec moins d'assurance
        self.declare_parameter('target_plate', '')
        self.declare_parameter('plate_threshold', 0.8)   # 0,8 tolère environ un caractère mal lu sur 9
        self.declare_parameter('max_ocr_per_frame', 2)   # Tesseract coûte ~0,1 s par lecture
        self.yolo = YoloOnnx(self.get_parameter('model_path').value, self.get_parameter('input_size').value,
                             self.get_parameter('yolo_conf').value)
        self.bridge, self.fx, self.cx = CvBridge(), None, None
        self.streak, self.last_bearing = 0, None
        self.frame, self.plate_cache = 0, {}
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.pub_cand = self.create_publisher(String, '/vision/target_candidate', 10)
        self.pub_ped = self.create_publisher(String, '/vision/pedestrians', 10)
        self.pub_obs = self.create_publisher(String, '/vision/observations', 10)   # toutes les voitures vues
        self.pub_dbg = self.create_publisher(Image, '/vision/debug', 1)
        self.get_logger().info('Vision prête')

    def on_info(self, msg):
        self.fx, self.cx = msg.k[0], msg.k[2]

    def bearing(self, u):
        """Angle (rad) de la colonne de pixels u, positif vers la gauche du robot."""
        return float(np.arctan2(self.cx - u, self.fx))

    def on_image(self, msg):
        if self.fx is None:
            return
        t0 = time.perf_counter()
        p = lambda n: self.get_parameter(n).value
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        none = ('', 'aucune', 'aucun', 'none', '-')
        color = '' if p('target_color').strip().lower() in none else p('target_color').strip().lower()
        plate_req = '' if p('target_plate').strip().lower() in none else p('target_plate').strip()
        cars, peds = [], []
        for x1, y1, x2, y2, conf, cls in self.yolo(img):
            name, u = COCO[cls], (x1 + x2) / 2
            if name == 'pieton':
                peds.append({'bearing': self.bearing(u), 'height_px': y2 - y1})
                continue
            w, h = x2 - x1, y2 - y1
            crop = img[max(0, int(y1 + .2 * h)):int(y2 - .2 * h), max(0, int(x1 + .2 * w)):int(x2 - .2 * w)]
            if crop.size == 0:
                continue
            sc = color_score(crop, color) if color else None
            cars.append({'box': (x1, y1, x2, y2), 'conf': conf, 'type': name, 'u': u, 'color_score': sc})
        # Lecture de plaque : la voiture la plus prometteuse à chaque image, les autres à tour de rôle.
        # Une lecture récente au même angle (< 1,5 s) est réutilisée : toutes les voitures gardent une info de plaque.
        if plate_req:
            now = time.monotonic()
            self.frame += 1
            cars.sort(key=lambda c: (c['color_score'] or 0.0, (c['box'][2] - c['box'][0])), reverse=True)
            rest = cars[1:]
            k = self.frame % len(rest) if rest else 0
            for c in (cars[:1] + rest[k:] + rest[:k])[:p('max_ocr_per_frame')]:
                x1, y1, x2, y2 = (int(v) for v in c['box'])
                lower = img[max(0, y1 + int(0.3 * (y2 - y1))):max(0, y2), max(0, x1):max(0, x2)]
                c['plate_read'] = read_plate_robust(lower) if lower.size else ''
                c['plate_score'] = plate_similarity(c['plate_read'], plate_req)
                self.plate_cache[round(self.bearing(c['u']) / 0.05)] = (now, c['plate_read'], c['plate_score'])
            for c in cars:
                if 'plate_score' not in c:
                    b = round(self.bearing(c['u']) / 0.05)
                    hits = [self.plate_cache[k2] for k2 in (b - 1, b, b + 1)
                            if k2 in self.plate_cache and now - self.plate_cache[k2][0] < 1.5]
                    if hits:
                        _, c['plate_read'], c['plate_score'] = max(hits, key=lambda h: h[0])
                c['ocr_tried'] = 'plate_score' in c
            self.plate_cache = {k2: v for k2, v in self.plate_cache.items() if now - v[0] < 3.0}
        best = None
        for c in cars:
            st = 1.0 if c['type'] == p('target_type') else 0.3
            sc, ps = c['color_score'], c.get('plate_score')
            if plate_req:
                ps = ps or 0.0
                score = 0.7 * ps + 0.2 * (sc if sc is not None else ps) + 0.1 * c['conf']
                if ps < p('plate_threshold'):
                    score = min(score, p('score_threshold') - 0.01)   # sans bonne plaque, jamais confirmé
            elif sc is not None:
                score = 0.6 * sc + 0.25 * st + 0.15 * c['conf']
            else:
                score = 0.7 * st + 0.3 * c['conf']
            x1, y1, x2, y2 = c['box']
            label = f"{c['type']} {score:.2f}" + (f" {c['plate_read']}" if c.get('plate_read') else '')
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label, (int(x1), max(12, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if best is None or score > best['score']:
                best = {'score': score, 'color_score': sc, 'type': c['type'], 'bearing': self.bearing(c['u']),
                        'bbox': [x1, y1, x2, y2], 'plate_read': c.get('plate_read', ''),
                        'plate_score': c.get('plate_score')}
        # Confirmation (EF6) : même candidat, même angle à ~8° près, sur plusieurs images consécutives
        if best and best['score'] >= p('score_threshold') and (
                self.last_bearing is None or abs(best['bearing'] - self.last_bearing) < 0.14):
            self.streak, self.last_bearing = self.streak + 1, best['bearing']
        else:
            self.streak, self.last_bearing = 0, None
        latency = (time.perf_counter() - t0) * 1000.0
        if best:
            best.update(confirmed=self.streak >= p('confirm_frames'), streak=self.streak, latency_ms=latency)
            self.pub_cand.publish(String(data=json.dumps(best)))
        self.pub_ped.publish(String(data=json.dumps(peds)))
        # Toutes les voitures de l'image, pour l'état des places et la vision active (car_locator)
        obs = [{'bearing': self.bearing(c['u']), 'type': c['type'], 'conf': c['conf'],
                'color_score': c['color_score'], 'plate_read': c.get('plate_read', ''),
                'plate_score': c.get('plate_score'), 'ocr_tried': c.get('ocr_tried', False)} for c in cars]
        self.pub_obs.publish(String(data=json.dumps(obs)))
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))


def main():
    rclpy.init()
    rclpy.spin(VisionNode())


if __name__ == '__main__':
    main()
