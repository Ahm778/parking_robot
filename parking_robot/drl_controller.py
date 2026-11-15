"""Contrôleur DRL (méthodes M2 et M3) : lit /scan + TF, applique la politique exportée, publie /cmd_vel à 5 Hz.

mode = drl    : vise directement le but reçu sur /drl/goal                (M2)
mode = hybrid : demande le chemin A* à Nav2 et vise 1 m plus loin dessus   (M3)
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener

from parking_robot.common import CmdVelPublisher, share_path
from parking_robot.drl_common import ACTIONS, DT, MAX_R, REL_ANGLES, ROBOT_R, NumpyPolicy, build_obs
from parking_robot.geometry import SPEED


class DRLController(Node):
    def __init__(self):
        super().__init__('drl_controller')
        self.declare_parameter('policy_path', share_path('weights', 'policy.npz'))
        self.declare_parameter('mode', 'hybrid')
        self.declare_parameter('lookahead', 1.0)
        self.policy = NumpyPolicy(self.get_parameter('policy_path').value)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.scan, self.goal, self.path, self.v, self.w = None, None, [], 0.0, 0.0
        self.last_progress, self.best_d, self.prev_rays = None, None, None
        self.align_steps, self.unstick, self.back_steps = 0, 0, 0
        self.aligning = False                             # dans la zone d'arrivée : on pivote jusqu'au bout
        self.hits = 0                                     # pas consécutifs en collision (dégagement)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, '/drl/goal', self.on_goal, 10)
        self.create_subscription(Empty, '/drl/cancel', self.on_cancel, 10)
        self.cmd = CmdVelPublisher(self)
        self.status = self.create_publisher(String, '/drl/status', 10)
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.create_timer(DT, self.loop)

    def on_scan(self, msg):
        self.scan = msg

    def on_cancel(self, _):
        self.goal, self.path = None, []
        self.cmd.send(0.0, 0.0)

    def on_goal(self, msg):
        self.goal, self.path, self.prev_rays, self.hits = msg, [], None, 0
        self.align_steps, self.unstick, self.back_steps, self.aligning = 0, 0, 0, False
        self.best_d, self.last_progress = None, self.get_clock().now()
        if self.get_parameter('mode').value == 'hybrid':
            g = ComputePathToPose.Goal()
            g.goal, g.use_start = msg, False
            self.planner.wait_for_server(timeout_sec=2.0)
            self.planner.send_goal_async(g).add_done_callback(self.on_path_accepted)

    def on_path_accepted(self, fut):
        handle = fut.result()
        if handle is not None and handle.accepted:
            handle.get_result_async().add_done_callback(self.on_path)

    def on_path(self, fut):
        poses = fut.result().result.path.poses
        self.path = [(p.pose.position.x, p.pose.position.y) for p in poses]

    def sectors(self):
        """24 secteurs de 15°, même ordre que REL_ANGLES (de -180° à +165° par rapport à l'avant)."""
        s, n = self.scan, len(self.scan.ranges)
        r = np.nan_to_num(np.asarray(s.ranges, dtype=float), nan=MAX_R, posinf=MAX_R)
        r[r < s.range_min] = MAX_R
        half = max(1, int(round(math.radians(7.5) / s.angle_increment)))
        idx = [int(round((a - s.angle_min) / s.angle_increment)) for a in REL_ANGLES]
        return np.array([min(r[(i + k) % n] for k in range(-half, half + 1)) for i in idx])

    def finish(self, status):
        self.cmd.send(0.0, 0.0)
        self.goal, self.path = None, []
        self.status.publish(String(data=status))

    def loop(self):
        if self.goal is None or self.scan is None:
            return
        try:
            t = self.tf.lookup_transform('map', 'base_footprint', Time())
        except Exception:  # noqa: BLE001
            return
        x, y = t.transform.translation.x, t.transform.translation.y
        q = t.transform.rotation
        th = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        d_goal = math.hypot(gx - x, gy - y)
        rays = self.sectors()
        # Zone d'arrivée avec hystérésis : on y entre à 0,3 m et on n'en sort qu'au-delà de 0,6 m.
        # Avant, le robot pivotait à 0,28 m, glissait à 0,30 m, la politique reprenait la main, l'éloignait,
        # puis il revenait : boucle « alignement » sans fin (S2 et S4 hybride du 10/11).
        if d_goal < 0.3 or (self.aligning and d_goal < 0.6):
            self.aligning = True
            q = self.goal.pose.orientation
            gyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            err = math.atan2(math.sin(gyaw - th), math.cos(gyaw - th))
            if abs(err) > 0.15 and self.align_steps < 75:     # pivoter vers l'orientation demandée (15 s max)
                self.align_steps += 1
                self.get_logger().info(f'alignement : erreur {err:.2f} rad, but à {d_goal:.2f} m',
                                       throttle_duration_sec=2.0)
                self.cmd.send(0.0, math.copysign(max(0.25, min(0.8, 2.0 * abs(err))), err) * min(SPEED, 1.5))
                return
            return self.finish('arrived')
        self.aligning = False
        if self.back_steps > 0:                               # manœuvre de dégagement en cours
            self.back_steps -= 1
            if rays[[23, 0, 1]].min() > 0.35:
                self.cmd.send(-0.08, 0.0)
            else:
                self.cmd.send(0.0, 0.8 if rays[:12].min() < rays[12:].min() else -0.8)
            return
        if rays.min() < ROBOT_R * 0.75:
            self.hits += 1
            if self.hits > 15:                            # 3 s de manœuvre sans succès : on abandonne
                return self.finish('collision')
            if rays[[23, 0, 1]].min() > 0.35:             # arrière dégagé : reculer
                self.cmd.send(-0.08, 0.0)
            else:                                         # sinon pivoter vers le côté le plus dégagé
                self.cmd.send(0.0, 0.8 if rays[:12].min() < rays[12:].min() else -0.8)
            return
        self.hits = 0
        # pas de progrès pendant 30 s -> bloqué
        now = self.get_clock().now()
        if self.best_d is None or d_goal < self.best_d - 0.1:
            self.best_d, self.last_progress = d_goal, now
        elif (now - self.last_progress).nanoseconds * 1e-9 > 20.0:
            if self.unstick >= 2:
                return self.finish('stuck')
            self.unstick += 1                                 # arrêt, recul 2 s, puis nouveau calcul
            self.back_steps = 10
            self.get_logger().info(f'bloqué : recul puis nouveau calcul (essai {self.unstick}/2), '
                                   f'but à {d_goal:.2f} m')
            self.last_progress = now
            self.cmd.send(0.0, 0.0)
            return
        # point visé : but final (M2) ou point du chemin A* situé lookahead plus loin (M3)
        tx, ty = gx, gy
        if self.path:
            k = min(range(len(self.path)), key=lambda i: math.hypot(self.path[i][0] - x, self.path[i][1] - y))
            self.path = self.path[k:]                         # oublie les points déjà dépassés
            la = self.get_parameter('lookahead').value
            tx, ty = next(((px, py) for px, py in self.path if math.hypot(px - x, py - y) >= la), (gx, gy))
        ang = math.atan2(ty - y, tx - x) - th
        ang = math.atan2(math.sin(ang), math.cos(ang))
        prev = rays if self.prev_rays is None else self.prev_rays
        self.prev_rays = rays
        a = self.policy.act(build_obs(rays, prev, math.hypot(tx - x, ty - y), ang, self.v, self.w))
        self.v, self.w = ACTIONS[a]
        front = rays[9:16].min()                          # cône avant ±45° (indice 12 = droit devant)
        if front < 0.45:                                  # trop près : on n'avance plus, rotation seulement
            self.get_logger().info(f'bouclier : obstacle à {front:.2f} m devant, '
                                   f'but à {d_goal:.2f} m', throttle_duration_sec=2.0)
            self.v = 0.0
            if self.w == 0.0:
                self.w = 0.8 if rays[9:12].min() < rays[13:16].min() else -0.8
        elif front < 0.80:
            self.v = min(self.v, 0.08)
        # Mode démo : rapide quand la voie devant est libre, puis ralentit progressivement en approchant
        # d'un obstacle ou d'une voiture (vitesse normale à moins de 0,6 m). Rotation jamais accélérée.
        # En mode normal SPEED = 1 : k = 1, comportement identique à la campagne.
        k = 1.0 + (SPEED - 1.0) * min(1.0, max(0.0, (front - 0.6) / 1.4))
        self.cmd.send(self.v * k, self.w)


def main():
    rclpy.init()
    rclpy.spin(DRLController())


if __name__ == '__main__':
    main()
