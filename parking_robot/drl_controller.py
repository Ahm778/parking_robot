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
        self.last_progress, self.best_d = None, None
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
        self.goal, self.path = msg, []
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
        if d_goal < 0.3:
            return self.finish('arrived')
        if rays.min() < ROBOT_R * 0.75:
            return self.finish('collision')
        # pas de progrès pendant 30 s -> bloqué
        now = self.get_clock().now()
        if self.best_d is None or d_goal < self.best_d - 0.1:
            self.best_d, self.last_progress = d_goal, now
        elif (now - self.last_progress).nanoseconds * 1e-9 > 30.0:
            return self.finish('stuck')
        # point visé : but final (M2) ou point du chemin A* situé lookahead plus loin (M3)
        tx, ty = gx, gy
        if self.path:
            k = min(range(len(self.path)), key=lambda i: math.hypot(self.path[i][0] - x, self.path[i][1] - y))
            self.path = self.path[k:]                         # oublie les points déjà dépassés
            la = self.get_parameter('lookahead').value
            tx, ty = next(((px, py) for px, py in self.path if math.hypot(px - x, py - y) >= la), (gx, gy))
        ang = math.atan2(ty - y, tx - x) - th
        ang = math.atan2(math.sin(ang), math.cos(ang))
        a = self.policy.act(build_obs(rays, math.hypot(tx - x, ty - y), ang, self.v, self.w))
        self.v, self.w = ACTIONS[a]
        self.cmd.send(self.v, self.w)


def main():
    rclpy.init()
    rclpy.spin(DRLController())


if __name__ == '__main__':
    main()
