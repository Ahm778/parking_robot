"""Enregistre une ligne de métriques par essai dans un CSV (EF15). Seul nœud autorisé à lire la vérité terrain (gt_)."""
import csv
import json
import math
import os

import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from parking_robot.common import share_path

END_PREFIXES = ('Arrivé devant', 'Échec de la navigation', 'Véhicule non trouvé')


class MetricsLogger(Node):
    def __init__(self):
        super().__init__('metrics_logger')
        for name, default in [('scenario', 'S1'), ('method', 'classic'), ('seed', 0),
                              ('out', os.path.expanduser('~/results.csv')), ('contact_dist', 0.15)]:
            self.declare_parameter(name, default)
        with open(share_path('config', 'parking_slots.yaml')) as f:
            self.gt = yaml.safe_load(f)['gt_target_slot']
        self.t0 = self.t_found = self.last = self.found = None
        self.dist, self.collisions, self.touching, self.done = 0.0, 0, False, False
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(String, '/target_car', self.on_target, 10)
        self.create_subscription(String, '/mission/status', self.on_status, 10)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9          # temps simulé (use_sim_time:=true)

    def on_odom(self, m):
        p = m.pose.pose.position
        if self.t0 is None:
            self.t0 = self.now()
        if self.last is not None:
            self.dist += math.hypot(p.x - self.last[0], p.y - self.last[1])
        self.last = (p.x, p.y)

    def on_scan(self, m):
        near = min((r for r in m.ranges if math.isfinite(r) and r > m.range_min), default=99.0)
        close = near < self.get_parameter('contact_dist').value
        self.collisions += int(close and not self.touching)        # front montant = un contact
        self.touching = close

    def on_target(self, m):
        if self.found is None:
            self.found, self.t_found = json.loads(m.data)['id'], self.now()

    def on_status(self, m):
        if self.done or not m.data.startswith(END_PREFIXES) or self.t0 is None:
            return
        self.done = True
        g = lambda n: self.get_parameter(n).value
        row = {'scenario': g('scenario'), 'method': g('method'), 'seed': g('seed'),
               'gt_slot': self.gt, 'found_slot': self.found,
               'loc_success': self.found == self.gt,
               'false_positive': self.found is not None and self.found != self.gt,
               'false_negative': self.found is None,
               'time_to_find_s': round(self.t_found - self.t0, 2) if self.t_found else '',
               'total_time_s': round(self.now() - self.t0, 2),
               'distance_m': round(self.dist, 2),
               'collisions': self.collisions,
               'mission_success': m.data.startswith('Arrivé') and self.found == self.gt and self.collisions == 0}
        out = g('out')
        new = not os.path.exists(out)
        with open(out, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        self.get_logger().info(f'Essai enregistré : {row}')
        raise SystemExit


def main():
    rclpy.init()
    node = MetricsLogger()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
