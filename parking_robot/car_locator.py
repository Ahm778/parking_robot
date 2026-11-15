"""Place les voitures vues sur la carte et suit l'état de chaque place.

Entrées : /vision/target_candidate (meilleur candidat), /vision/observations (toutes les voitures), /scan, TF.
Sorties :
  /target_car          (std_msgs/String, JSON) : place de la voiture trouvée (après 3 votes concordants)
  /parking/slot_states (std_msgs/String, JSON) : état de chaque place (unseen, seen, mismatch, suspect, found)
  /parking/markers     (visualization_msgs/MarkerArray) : affichage RViz des places, plaques lues, voiture trouvée
Méthode : angle donné par la caméra + distance lidar dans cette direction -> point dans « map » -> place la plus proche.
"""
import json
import math

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (rend tf.transform utilisable sur PointStamped)
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from parking_robot.common import load_slots
from rclpy.time import Time
from parking_robot.geometry import AISLE_HALF, AISLES_Y, ROW_AISLE, SLOT_L, SLOT_W
from parking_robot.slot_tracker import COLORS, SlotTracker

NONE_WORDS = ('', 'aucune', 'aucun', 'none', '-')


def range_at(scan, bearing, half_window=3):
    """Médiane des distances lidar autour de l'angle demandé (rad, repère du lidar)."""
    n = len(scan.ranges)
    i = int(round((bearing - scan.angle_min) / scan.angle_increment))
    win = [scan.ranges[(i + k) % n] for k in range(-half_window, half_window + 1)]
    win = [r for r in win if math.isfinite(r) and scan.range_min < r < scan.range_max]
    return float(np.median(win)) if win else None


class CarLocator(Node):
    def __init__(self):
        super().__init__('car_locator')
        self.declare_parameter('min_votes', 3)
        self.declare_parameter('target_plate', 'aucune')
        self.declare_parameter('target_color', 'rouge')
        self.declare_parameter('plate_threshold', 0.8)
        self.slots = load_slots()
        plate_req = str(self.get_parameter('target_plate').value).strip().lower() not in NONE_WORDS
        color_req = str(self.get_parameter('target_color').value).strip().lower() not in NONE_WORDS
        self.tracker = SlotTracker(self.slots, plate_req, color_req, self.get_parameter('plate_threshold').value)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.scan, self.votes, self.found = None, {}, None
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(String, '/vision/target_candidate', self.on_candidate, 10)
        self.create_subscription(String, '/vision/observations', self.on_observations, 10)
        self.create_subscription(String, '/parking/inspected', self.on_inspected, 10)
        self.pub = self.create_publisher(String, '/target_car', 10)
        self.pub_states = self.create_publisher(String, '/parking/slot_states', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/parking/markers', 10)
        self.create_timer(0.5, self.publish_state)

    def on_scan(self, msg):
        self.scan = msg

    def project(self, bearing):
        """Angle caméra -> point (x, y) dans la carte, ou None."""
        if self.scan is None:
            return None
        r = range_at(self.scan, bearing)
        if r is None:
            return None                                     # rien à portée du lidar dans cette direction
        p = PointStamped()
        p.header.frame_id = self.scan.header.frame_id       # base_scan ; stamp à 0 = dernière TF connue
        p.point.x, p.point.y = r * math.cos(bearing), r * math.sin(bearing)
        try:
            q = self.tf.transform(p, 'map', timeout=Duration(seconds=0.2)).point
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'TF map indisponible : {e}', throttle_duration_sec=5.0)
            return None
        return q.x, q.y

    def nearest_here(self, x, y):
        """Place la plus proche du point vu, limitée aux 2 rangées qui bordent l'allée où roule le robot.

        Mode démo (plusieurs allées) : par une place vide, la caméra voit l'arrière des voitures de la rangée
        suivante ; sans ce filtre elles devenaient « suspectes » et le robot changeait d'allée pour rien.
        Mode normal (une seule allée) : identique à avant."""
        if len(AISLES_Y) == 1:
            return self.tracker.nearest(x, y)
        try:
            ry = self.tf.lookup_transform('map', 'base_footprint', Time()).transform.translation.y
        except Exception:  # noqa: BLE001
            return None
        aisle = min(AISLES_Y, key=lambda a: abs(a - ry))
        if abs(aisle - ry) > AISLE_HALF + 0.5:
            return None                                 # dans la voie périphérique : on n'attribue rien
        here = [s for s in self.tracker.slots.values() if ROW_AISLE[s['row']][0] == aisle]
        best = min(here, key=lambda s: (s['x'] - x) ** 2 + (s['y'] - y) ** 2)
        d = ((best['x'] - x) ** 2 + (best['y'] - y) ** 2) ** 0.5
        return best['id'] if d <= self.tracker.max_dist else None

    def on_candidate(self, msg):
        c = json.loads(msg.data)
        if not c['confirmed'] or self.found:
            return
        xy = self.project(c['bearing'])
        if xy is None:
            return
        sid = self.nearest_here(*xy)
        if sid is None:
            return
        self.votes[sid] = self.votes.get(sid, 0) + 1
        best = max(self.votes, key=self.votes.get)
        if self.votes[best] >= self.get_parameter('min_votes').value:
            self.found = best
            self.tracker.set_found(best)
            s = next(s for s in self.slots if s['id'] == best)
            self.pub.publish(String(data=json.dumps({**s, 'votes': self.votes[best], 'hit': list(xy)})))
        if self.found:                                       # republier pour le gestionnaire de mission
            s = next(s for s in self.slots if s['id'] == self.found)
            self.pub.publish(String(data=json.dumps({**s, 'votes': self.votes[self.found]})))

    def on_observations(self, msg):
        for o in json.loads(msg.data):
            xy = self.project(o['bearing'])
            sid = self.nearest_here(*xy) if xy else None
            if sid:
                self.tracker.observe(sid, o)

    def on_inspected(self, msg):
        """Le gestionnaire de mission a regardé cette place de près sans conclure."""
        if msg.data in self.tracker.slots:
            self.tracker.mark_inspected_unreadable(msg.data)

    def publish_state(self):
        summary = self.tracker.summary()
        self.pub_states.publish(String(data=json.dumps(summary)))
        self.pub_markers.publish(self.markers(summary))

    def markers(self, summary):
        arr, stamp = MarkerArray(), self.get_clock().now().to_msg()
        for i, s in enumerate(summary.values()):
            box = Marker()
            box.header.frame_id, box.header.stamp = 'map', stamp
            box.ns, box.id, box.type, box.action = 'slots', i, Marker.CUBE, Marker.ADD
            box.pose.position.x, box.pose.position.y, box.pose.position.z = float(s['x']), float(s['y']), 0.02
            box.pose.orientation.w = 1.0
            box.scale.x, box.scale.y, box.scale.z = SLOT_W - 0.3, SLOT_L - 0.3, 0.04
            box.color.r, box.color.g, box.color.b, box.color.a = COLORS[s['state']]
            arr.markers.append(box)
            txt = Marker()
            txt.header.frame_id, txt.header.stamp = 'map', stamp
            txt.ns, txt.id, txt.type, txt.action = 'labels', i, Marker.TEXT_VIEW_FACING, Marker.ADD
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(s['x']), float(s['y']), 2.2
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.45
            txt.color.r, txt.color.g, txt.color.b, txt.color.a = 1.0, 1.0, 1.0, 1.0
            txt.text = s['id'] + (f"  {s['plate']}" if s['plate'] else '')
            arr.markers.append(txt)
        if self.found:
            s = summary[self.found]
            m = Marker()
            m.header.frame_id, m.header.stamp = 'map', stamp
            m.ns, m.id, m.type, m.action = 'found', 0, Marker.ARROW, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(s['x']), float(s['y']), 4.2
            m.pose.orientation.y, m.pose.orientation.w = math.sin(math.pi / 4), math.cos(math.pi / 4)  # vers le bas
            m.scale.x, m.scale.y, m.scale.z = 1.5, 0.35, 0.35
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 1.0
            arr.markers.append(m)
        return arr


def main():
    rclpy.init()
    rclpy.spin(CarLocator())


if __name__ == '__main__':
    main()
