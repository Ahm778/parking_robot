"""Scénario S4 « chemin bloqué » : après un délai, place l'obstacle « blocker » au milieu de l'allée.

Piste 22.04 : service ROS /gazebo/set_entity_state
Piste 26.04 : service Gazebo /world/parking/set_pose (outil en ligne de commande « gz service »)
"""
import subprocess

import rclpy
from rclpy.node import Node

from parking_robot.common import IS_CLASSIC
from parking_robot.geometry import BLOCKER


class ScenarioEvents(Node):
    def __init__(self):
        super().__init__('scenario_events')
        self.declare_parameter('delay_s', 20.0)     # temps simulé avant l'apparition de l'obstacle
        self.declare_parameter('x', BLOCKER[0])
        self.declare_parameter('y', BLOCKER[1])
        self.done = False
        if IS_CLASSIC:
            from gazebo_msgs.srv import SetEntityState
            self.Req = SetEntityState.Request
            self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.t0 = None
        self.create_timer(0.5, self.tick)

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        if self.done or now - self.t0 < self.get_parameter('delay_s').value:
            return
        x, y = self.get_parameter('x').value, self.get_parameter('y').value
        if IS_CLASSIC:
            if not self.cli.service_is_ready():
                return
            req = self.Req()
            req.state.name = 'blocker'
            req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = x, y, 0.5
            req.state.pose.orientation.w = 1.0
            req.state.reference_frame = 'world'
            self.cli.call_async(req)
        else:
            subprocess.run(['gz', 'service', '-s', '/world/parking/set_pose',
                            '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                            '--timeout', '2000',
                            '--req', f'name: "blocker", position: {{x: {x}, y: {y}, z: 0.5}}'],
                           check=False)
        self.done = True
        self.get_logger().info(f'Obstacle placé en ({x}, {y})')


def main():
    rclpy.init()
    rclpy.spin(ScenarioEvents())


if __name__ == '__main__':
    main()
