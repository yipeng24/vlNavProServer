from typing import Optional
from dataclasses import dataclass

from sensor_msgs.msg import Joy

@dataclass
class joy_enum:
    A,B,X,Y,LB,RB = 0,1,2,3,4,5
    BACK,START,POWER,L_STICK,R_STICK = 6,7,8,9,10
    horizontal_L,vertical_L = 0,1
    horizontal_R,vertical_R = 2,3
    LT,RT = 4,5


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
        self._trans_snapshot_triggered = False

        self._call_vlm_button = joy_enum.LB  # LB 键
        self._call_vlm_button_pressed = False
        self._trans_vlm_triggered = False

        self._ensure_nav_button = joy_enum.Y 
        self._ensure_nav_button_pressed = False
        self._ensure_nav_triggered = False

    def _button_is_pressed(self, joy: Optional[Joy], button_id: int) -> bool:
        if joy is None:
            return False
        if button_id < 0 or button_id >= len(joy.buttons):
            return False
        return joy.buttons[button_id] == 1

    def joy_update(self, joy: Joy):
        self._joy_state = joy

        # Hold A to forward Nav2's /cmd_vel_nav to the robot /cmd_vel output.
        self.nav_button_pressed = self._button_is_pressed(self._joy_state, self.nav_hold_button)
        self._trans_use_nav = self.nav_button_pressed

        self.snapshot_button_pressed = self._button_pressed_edge(joy_enum.B)
        if self.snapshot_button_pressed and not self._trans_snapshot_triggered:
            self._trans_snapshot_triggered = True

        self._call_vlm_button_pressed = self._button_pressed_edge(self._call_vlm_button)
        if self._call_vlm_button_pressed and not self._trans_vlm_triggered:
            self._trans_vlm_triggered = True

        self._ensure_nav_button_pressed = self._button_pressed_edge(self._ensure_nav_button)
        if self._ensure_nav_button_pressed and not self._ensure_nav_triggered:
            self._ensure_nav_triggered = True

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

        linear_velocity = -linear_input * self.scale_linear
        angular_velocity = -angular_input * self.scale_angular

        return linear_velocity, angular_velocity


    def _button_hold_pressed(self, button_id: int) -> bool:
        if self._joy_state is None or self._lastest_joy_state is None:
            return False
        return (
            self._button_is_pressed(self._joy_state, button_id) and
            self._button_is_pressed(self._lastest_joy_state, button_id)
        )


    def _button_pressed_edge(self, button_id: int) -> bool:
        if self._joy_state is None:
            return False
        if self._lastest_joy_state is None:
            return self._button_is_pressed(self._joy_state, button_id)
        return (
            self._button_is_pressed(self._joy_state, button_id) and
            not self._button_is_pressed(self._lastest_joy_state, button_id)
        )
