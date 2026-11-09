"""Fonctions partagées par les nœuds : piste de simulation, commande de vitesse, fichiers du package.

Piste 22.04 : ROS 2 Humble + Gazebo Classic -> /cmd_vel en geometry_msgs/Twist
Piste 26.04 : ROS 2 Lyrical + Gazebo Jetty  -> /cmd_vel en geometry_msgs/TwistStamped
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, TwistStamped

ROS_DISTRO = os.environ.get('ROS_DISTRO', '')
IS_CLASSIC = ROS_DISTRO == 'humble'          # Humble -> Gazebo Classic ; sinon Gazebo moderne (gz sim)


def share_path(*parts):
    return os.path.join(get_package_share_directory('parking_robot'), *parts)


def load_slots():
    """Géométrie des places (connue du robot). Les champs gt_ ne sont PAS renvoyés."""
    with open(share_path('config', 'parking_slots.yaml')) as f:
        data = yaml.safe_load(f)
    keys = ('id', 'row', 'number', 'x', 'y')
    return [{k: s[k] for k in keys} for s in data['slots']]


class CmdVelPublisher:
    """Publie /cmd_vel avec le bon type selon la piste (Twist ou TwistStamped)."""

    def __init__(self, node, topic='/cmd_vel'):
        self.node = node
        self.stamped = not IS_CLASSIC
        self.pub = node.create_publisher(TwistStamped if self.stamped else Twist, topic, 10)

    def send(self, v, w):
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        else:
            msg = Twist()
            msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)
