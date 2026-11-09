"""Ralentit Nav2 près des piétons (EF11) : vision (c'est un piéton) + lidar (à quelle distance) -> /speed_limit.

Agit sur la méthode M1 (Nav2). Pour M2/M3, l'agent DRL gère lui-même la proximité (pénalité à l'entraînement).
"""
import json
import math

import rclpy
from nav2_msgs.msg import SpeedLimit
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class PedestrianGuard(Node):
    def __init__(self):
        super().__init__('pedestrian_guard')
        self.declare_parameter('slow_dist', 2.0)    # m : en dessous -> 30 % de la vitesse max
        self.declare_parameter('crawl_dist', 1.0)   # m : en dessous -> 5 %
        self.scan = None
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(String, '/vision/pedestrians', self.on_peds, 10)
        self.pub = self.create_publisher(SpeedLimit, '/speed_limit', 10)

    def on_scan(self, msg):
        self.scan = msg

    def dist(self, bearing):
        s, n = self.scan, len(self.scan.ranges)
        i = int(round((bearing - s.angle_min) / s.angle_increment))
        r = [s.ranges[(i + k) % n] for k in range(-4, 5)]
        r = [x for x in r if math.isfinite(x) and s.range_min < x < s.range_max]
        return min(r) if r else float('inf')

    def on_peds(self, msg):
        if self.scan is None:
            return
        d = min([self.dist(p['bearing']) for p in json.loads(msg.data)] or [float('inf')])
        if d < self.get_parameter('crawl_dist').value:
            pct = 5.0
        elif d < self.get_parameter('slow_dist').value:
            pct = 30.0
        else:
            pct = 100.0            # 100 % = pas de limitation (attention : 0 signifie aussi « pas de limite »)
        out = SpeedLimit()
        out.percentage, out.speed_limit = True, pct
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(PedestrianGuard())


if __name__ == '__main__':
    main()
