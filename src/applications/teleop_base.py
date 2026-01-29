from typing import Optional
from dataclasses import dataclass

from sensor_msgs.msg import Joy

@dataclass
class joy_enum:
    A=0,B=1,X=2,Y=3,LB=4,RB=5,
    BACK=6,START=7,POWER=8,L_STICK=9,R_STICK=10,
    horizontal_L=0,vertical_L=1,
    LT=2,horizontal_R=3,vertical_R=4,
    RT=5


class teleop_base:
    def __init__(self):
        self._joy_state: Optional[Joy] = None
        self._lastest_joy_state: Optional[Joy] = None
        # 轴映射：
        self.axis_linear = joy_enum.vertical_L
        self.axis_angular = joy_enum.horizontal_L
        self.scale_linear = 0.4      # m/s
        self.scale_angular = 1.5     # rad/s
        self.deadzone = 0.05

        self.nav_hold_button = joy_enum.A     # A 键
        self.nav_button_pressed = False
        self._trans_use_nav = False

        self.snapshot_button = joy_enum.B     # B 键
        self.snapshot_button_pressed = False
        self._trans_snapshot_trigger = False


    def joy_update(self, joy: Joy):
        self._joy_state = joy

        self.nav_button_pressed = self._button_hold_pressed(joy_enum.A)

        self.snapshot_button_pressed = self._button_pressed_edge(joy_enum.B)
        if self.snapshot_button_pressed and not self._trans_snapshot_trigger:
            self._trans_snapshot_trigger = True

        self._lastest_joy_state = self._joy_state


    def calc_cmd_vel(self) -> tuple[float, float]:
        if self._joy_state is None:
            return 0.0, 0.0

        linear_input = self._joy_state.axes[self.axis_linear]
        angular_input = self._joy_state.axes[self.axis_angular]

        # apply deadzone
        if abs(linear_input) < self.deadzone:
            linear_input = 0.0
        if abs(angular_input) < self.deadzone:
            angular_input = 0.0

        linear_velocity = linear_input * self.scale_linear
        angular_velocity = angular_input * self.scale_angular

        return linear_velocity, angular_velocity


    def _button_hold_pressed(self, button_id: int) -> bool:
        if self._joy_state is None or self._lastest_joy_state is None:
            return False
        return (self._joy_state.buttons[button_id] == 1 and
                self._lastest_joy_state.buttons[button_id] == 1)


    def _button_pressed_edge(self, button_id: int) -> bool:
        if self._joy_state is None or self._lastest_joy_state is None:
            return False
        return (self._joy_state.buttons[button_id] == 1 and
                self._lastest_joy_state.buttons[button_id] == 0)
