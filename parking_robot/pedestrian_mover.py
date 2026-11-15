"""Fait traverser l'allée au piéton « ped_1 » en va-et-vient (scénarios S2, S3).

Piste 22.04 (Gazebo Classic) : téléportation à 20 Hz via le service /gazebo/set_entity_state
Piste 26.04 (Gazebo Jetty)   : vitesse envoyée au plugin VelocityControl via /model/ped_1/cmd_vel (pont ros_gz)
Comme un vrai piéton, il s'arrête quand le robot est à moins de yield_dist (0 = ne s'arrête jamais).
Même règle que l'environnement d'entraînement DRL (drl_env.py, YIELD_DIST).
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from parking_robot.common import IS_CLASSIC
from parking_robot.geometry import PED


class PedestrianMover(Node):
    def __init__(self):
        super().__init__('pedestrian_mover')
        self.declare_parameter('name', 'ped_1')
        self.declare_parameter('x', PED[0])         # où il traverse : entre deux colonnes d'observation (m)
        self.declare_parameter('y_min', PED[1])     # doit correspondre à la pose initiale dans le monde
        self.declare_parameter('y_max', PED[2])
        self.declare_parameter('speed', 0.5)      # m/s
        self.declare_parameter('yield_dist', 0.8)  # m ; 0 = piéton « aveugle » (test de stress)
        self.t, self.y = 0.0, None
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        if IS_CLASSIC:
            from gazebo_msgs.srv import SetEntityState
            self.Req = SetEntityState.Request
            self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        else:
            name = self.get_parameter('name').value
            self.pub = self.create_publisher(Twist, f'/model/{name}/cmd_vel', 10)
        self.create_timer(0.05, self.step)

    def triangle(self):
        """Position y et sens (+1/-1) d'un aller-retour à vitesse constante entre y_min et y_max."""
        p = lambda n: self.get_parameter(n).value
        span = p('y_max') - p('y_min')
        d = (p('speed') * self.t) % (2 * span)
        return (p('y_min') + d, 1.0) if d < span else (p('y_max') - (d - span), -1.0)

    def robot_close(self):
        d = float(self.get_parameter('yield_dist').value)
        if d <= 0.0 or self.y is None:
            return False
        try:
            t = self.tf.lookup_transform('map', 'base_footprint', Time())
        except Exception:  # noqa: BLE001
            return False
        rx, ry = t.transform.translation.x, t.transform.translation.y
        return math.hypot(rx - float(self.get_parameter('x').value), ry - self.y) < d

    def step(self):
        waiting = self.robot_close()
        if not waiting:
            self.t += 0.05
        y, direction = self.triangle()
        self.y = y
        if IS_CLASSIC:
            if not self.cli.service_is_ready():
                return
            req = self.Req()
            req.state.name = self.get_parameter('name').value
            req.state.pose.position.x = float(self.get_parameter('x').value)
            req.state.pose.position.y = float(y)
            req.state.pose.orientation.w = 1.0
            req.state.reference_frame = 'world'
            self.cli.call_async(req)
        else:
            msg = Twist()
            msg.linear.y = 0.0 if waiting else direction * float(self.get_parameter('speed').value)
            self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(PedestrianMover())


if __name__ == '__main__':
    main()
