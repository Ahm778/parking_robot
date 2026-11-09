"""Gestionnaire de mission : exploration des rangées -> véhicule trouvé -> approche -> message.

Vision active : après chaque arrêt, si une place est « suspecte » (voiture compatible mais plaque illisible
d'ici), le robot va se placer juste devant elle pour relire la plaque, puis reprend la recherche.

Paramètre method :
  classic : Nav2 (A* + DWB) se charge du trajet          -> méthode M1
  drl     : drl_controller vise directement le but       -> méthode M2
  hybrid  : drl_controller suit le chemin A* de Nav2     -> méthode M3
"""
import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_msgs.msg import Empty, String

from parking_robot.common import load_slots


class Mission(BasicNavigator):
    def __init__(self):
        super().__init__(node_name='mission_manager')
        self.declare_parameter('method', 'classic')
        self.declare_parameter('pause_s', 3.0)          # arrêt à chaque point pour laisser la vision confirmer
        self.declare_parameter('look_y', 1.0)           # décalage vers la rangée observée : plaques à ~2,5 m
        self.declare_parameter('inspect_y', 1.6)        # vision active : plaques à ~1,8 m
        self.declare_parameter('active_vision', True)
        self.declare_parameter('approach_y', 2.2)       # distance au centre de l'allée du point d'approche (m)
        self.method = self.get_parameter('method').value
        self.found, self.drl_status, self.slot_states, self.inspected = {}, None, {}, set()
        self.create_subscription(String, '/parking/slot_states',
                                 lambda m: setattr(self, 'slot_states', json.loads(m.data)), 10)
        self.inspected_pub = self.create_publisher(String, '/parking/inspected', 10)
        self.create_subscription(String, '/target_car', lambda m: self.found.update(json.loads(m.data)), 10)
        self.create_subscription(String, '/drl/status', self.on_drl_status, 10)
        self.status_pub = self.create_publisher(String, '/mission/status', 10)
        self.drl_goal_pub = self.create_publisher(PoseStamped, '/drl/goal', 10)
        self.drl_cancel_pub = self.create_publisher(Empty, '/drl/cancel', 10)

    # ---------- outils ----------
    def say(self, txt):
        self.status_pub.publish(String(data=txt))
        self.get_logger().info(txt)

    def on_drl_status(self, msg):
        self.drl_status = msg.data

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def wait(self, seconds, stop_if_found=True):
        end = self.now_s() + seconds
        while rclpy.ok() and self.now_s() < end and not (stop_if_found and self.found):
            rclpy.spin_once(self, timeout_sec=0.1)

    def pose(self, x, y, yaw):
        p = PoseStamped()
        p.header.frame_id, p.header.stamp = 'map', self.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return p

    def go_to(self, x, y, yaw, stop_if_found=True, timeout_s=180.0):
        """Va au point. Renvoie 'ok', 'found' (véhicule trouvé en route) ou 'fail'."""
        goal = self.pose(x, y, yaw)
        t0 = self.now_s()
        if self.method == 'classic':
            self.goToPose(goal)
            while not self.isTaskComplete():
                if stop_if_found and self.found:
                    self.cancelTask()
                    return 'found'
                if self.now_s() - t0 > timeout_s:
                    self.cancelTask()
                    return 'fail'
            return 'ok' if self.getResult() == TaskResult.SUCCEEDED else 'fail'
        # drl / hybrid
        self.drl_status = None
        self.drl_goal_pub.publish(goal)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if stop_if_found and self.found:
                self.drl_cancel_pub.publish(Empty())
                return 'found'
            if self.drl_status == 'arrived':
                return 'ok'
            if self.drl_status in ('collision', 'stuck') or self.now_s() - t0 > timeout_s:
                self.drl_cancel_pub.publish(Empty())
                return 'fail'
        return 'fail'

    def inspect_suspects(self, here_x, only_column=True):
        """Vision active : va relire de près les plaques des places suspectes, la plus proche d'abord.

        Pendant l'exploration, seulement la colonne où se trouve le robot (les autres seront vues de face plus tard) ;
        à la fin, toutes celles qui restent.
        """
        if not self.get_parameter('active_vision').value:
            return
        while not self.found:
            todo = [s for sid, s in self.slot_states.items() if s['state'] == 'suspect' and sid not in self.inspected
                    and (not only_column or abs(s['x'] - here_x) < 1.5)]
            if not todo:
                return
            s = min(todo, key=lambda s: abs(s['x'] - here_x))
            side = 1.0 if s['y'] > 0 else -1.0
            self.say(f"Place {s['id']} : plaque illisible d'ici, je m'approche")
            res = self.go_to(s['x'], side * self.get_parameter('inspect_y').value, side * math.pi / 2)
            if res == 'found' or self.found:
                return
            self.wait(self.get_parameter('pause_s').value)
            if self.found:
                return
            self.inspected.add(s['id'])
            self.inspected_pub.publish(String(data=s['id']))
            here_x = s['x']

    # ---------- mission ----------
    def run(self):
        # Pas d'AMCL dans notre config : on attend map_server (nœud lifecycle présent sur Humble et Lyrical)
        self.waitUntilNav2Active(localizer='map_server')
        xs = sorted({s['x'] for s in load_slots()})
        ly = self.get_parameter('look_y').value
        # pour chaque colonne : face à la rangée A (un peu de son côté), puis face à la rangée B
        waypoints = [wp for x in xs for wp in ((x, ly, math.pi / 2), (x, -ly, -math.pi / 2))]
        t0 = self.now_s()
        self.say(f'Recherche démarrée (méthode {self.method})')
        for x, y, yaw in waypoints:
            if self.go_to(x, y, yaw) == 'found':
                break
            self.wait(self.get_parameter('pause_s').value)
            if self.found:
                break
            self.inspect_suspects(x)
            if self.found:
                break
        if not self.found:                                   # dernière passe : suspects restants
            self.inspect_suspects(0.0, only_column=False)
        if not self.found:
            self.say('Véhicule non trouvé')
            return
        f = self.found
        self.say(f"Véhicule trouvé : rangée {f['row']}, place {f['number']} ({self.now_s() - t0:.1f} s)")
        side = 1.0 if f['y'] > 0 else -1.0
        res = self.go_to(f['x'], side * self.get_parameter('approach_y').value, side * math.pi / 2,
                         stop_if_found=False)
        self.say('Arrivé devant votre véhicule' if res == 'ok' else 'Échec de la navigation')


def main():
    rclpy.init()
    node = Mission()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
