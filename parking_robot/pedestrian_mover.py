"""Fait traverser l'allée au piéton « ped_1 » en va-et-vient (scénarios S2, S3).

Piste 22.04 (Gazebo Classic) : téléportation à 20 Hz via le service /gazebo/set_entity_state
Piste 26.04 (Gazebo Jetty)   : vitesse envoyée au plugin VelocityControl via /model/ped_1/cmd_vel (pont ros_gz)
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from parking_robot.common import IS_CLASSIC


class PedestrianMover(Node):
    def __init__(self):
        super().__init__('pedestrian_mover')
        self.declare_parameter('name', 'ped_1')
        self.declare_parameter('x', 0.0)          # colonne où il traverse (m)
        self.declare_parameter('y_min', -2.5)     # doit correspondre à la pose initiale dans le monde
        self.declare_parameter('y_max', 2.5)
        self.declare_parameter('speed', 0.5)      # m/s
        self.t = 0.0
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

    def step(self):
        self.t += 0.05
        y, direction = self.triangle()
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
            msg.linear.y = direction * float(self.get_parameter('speed').value)
            self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(PedestrianMover())


if __name__ == '__main__':
    main()
