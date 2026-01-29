from dataclasses import dataclass
from enum import Enum, auto
import os
import time
from image_pool.buffer import ImageRingBuffer

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

import rclpy
@dataclass
class ILGP_State(Enum):
    INIT = auto(),
    WAIT_TRIGGER = auto(),
    PLAN_TRAJ = auto(),
    Moving = auto(),

class singleTh_container(Node):
    def __init__(self):
        super().__init__("singleTh_container")
        self._teleop_timer = self.create_timer(0.1, self._teleop_timer_callback)
        self.state: ILGP_State = ILGP_State.INIT
        
        self._sub_joy = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 10)
        self._sub_nav = self.create_subscription(Twist, self.cmd_vel_nav_topic, self.nav_callback, 10)
        
    
    def _teleop_timer_callback(self):
        pass

    def joy_callback(self, msg: Joy):
        pass

    def nav_callback(self, msg: Twist):
        pass