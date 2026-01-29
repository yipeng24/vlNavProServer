#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

from kobuki_ros_interfaces.msg import MotorPower

from image_pool.buffer import ImageRingBuffer

class XBotTeleop(Node):
    def __init__(self, ring : ImageRingBuffer = None):
        super().__init__('xbot_teleop')

        # =====topic config=====
        self.joy_topic = '/joy'
        self.cmd_vel_joy_topic = '/cmd_vel_joy'
        self.cmd_vel_nav_topic = '/cmd_vel_nav'
        self.cmd_vel_out_topic = '/cmd_vel'
        self.motor_power_topic = 'motor_power'  # kobuki_node 通常就是这个

        self.joy_cmd = Twist()       # 手柄算出来的速度
        self.nav_cmd = Twist()       # 订阅到的 Nav2 速度
        # ===hardware_config 配置==================
        self._latest_joy: Optional[Joy] = None
        # 轴映射：
        self.axis_linear = 1
        self.axis_angular = 0

        # 方向反了就改这里k
        self.invert_linear = False
        self.invert_angular = False

        # 速度缩放
        self.scale_linear = 0.4      # m/s
        self.scale_angular = 1.5     # rad/s
        self.deadzone = 0.05

        # 发布频率（Hz）
        self.publish_rate_hz = 10.0

        # 电机上电/断电按钮（不需要就设为 -1）
        self.enable_button = 3       # 例如 Y
        self.disable_button = 1      # 例如 B

        # 启动时是否自动上电
        self.enable_on_start = True

        #===nav2 配置==================
        self.nav_hold_button = 0     # A 键
        self.require_nav_hold = True # True=按住A才用Nav2速度（更安全）  

        self.power_status = False
        self.last_zero_vel_sent = True

        self._last_buttons = None
        self._last_axes = None
        self.nav_button_pressed = False

        #= snapshot config ==============
        self.enable_snapshot = True
        self.snapshot_button = 2          # 默认 X=2（不对就改）

        #=====vlm config ======
        self.enable_vlm = True
        self.vlm_button = 5       # B(示例)
        self.vlm_k = 4
        #=====ring buffer=====
        self.ring = ring  # type: ImageRingBuffer
        self.save_dir = '/home/yipeng/image_bridge_saved'
        os.makedirs(self.save_dir, exist_ok=True)
        # =========================
        # Pub/Sub（原有）
        # =========================
        self.joy_pub = self.create_publisher(Twist, self.cmd_vel_joy_topic, 10)
        self.out_pub = self.create_publisher(Twist, self.cmd_vel_out_topic, 10)

        motor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.motor_pub = self.create_publisher(MotorPower, self.motor_power_topic, motor_qos)

        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 10)
        self.nav_sub = self.create_subscription(Twist, self.cmd_vel_nav_topic, self.nav_callback, 10)
        
        if self.enable_on_start:
            self.enable()
        # Timer: 定时发布
        period = 1.0 / max(1e-6, float(self.publish_rate_hz))
        self.timer = self.create_timer(period, self.spin_publish)

        self.get_logger().info(
            "XBotTeleop started (shared ring = {}). "
            "snapshot_button={}, vlm_button={}, nav={}".format(
                "YES" if self.ring is not None else "NO",
                self.snapshot_button,
                self.vlm_button,
                self.nav_hold_button
            )
        )

    @staticmethod
    def _apply_deadzone(v: float, dz: float) -> float:
        return 0.0 if abs(v) < dz else v


    def _button_pressed_edge(self, buttons, idx: int) -> bool:
        if idx < 0:
            return False
        if self._last_buttons is None:
            return False
        if idx >= len(buttons) or idx >= len(self._last_buttons):
            return False
        return (self._last_buttons[idx] == 0) and (buttons[idx] == 1)


    def _nav_hold_pressed(self, buttons) -> bool:
        if not self.require_nav_hold:
            return True
        if self.nav_hold_button < 0:
            return True
        if self.nav_hold_button >= len(buttons):
            return False
        return buttons[self.nav_hold_button] == 1


    def _safe_get_latest(self, n: int):
        try:
            return self.ring.get_latest(n)
        except Exception as e:
            self.get_logger().error(f"ring.get_latest({n}) failed: {e}")
            return []
        

    def save_current_frame(self):
        # 存在检查
        if self.ring is None:
            self.get_logger().warn("No shared ring -> cannot save frame.")
            return
        info = self.ring.save_latest(out_dir=self.save_dir)
        if info.get("ok", False):
            self.get_logger().info(f"📸 {info.get('msg')} rgb={info.get('rgb_path')}")
        else:
            self.get_logger().warn(f"📸 Save failed: {info.get('msg')}")


    def run_vlm_on_last_k(self):
        self.get_logger().info("VLM triggered (not implemented).")
        if self.ring is None:
            self.get_logger().warn("No shared ring -> cannot fetch last K frames for VLM.")
            return
        

    def nav_callback(self, msg: Twist):
        self.nav_cmd = msg


    def joy_callback(self, msg: Joy):
        self._latest_joy = msg
        axes = msg.axes
        buttons = msg.buttons

        # A键状态（按住用Nav2）
        self.nav_button_pressed = self._nav_hold_pressed(buttons)

        if self.enable_snapshot and self._button_pressed_edge(buttons, self.snapshot_button):
            self.save_current_frame()
        if self.enable_vlm and self._button_pressed_edge(buttons, self.vlm_button):
            self.run_vlm_on_last_k()
        # enable/disable：按下沿触发
        if self._button_pressed_edge(buttons, self.enable_button):
            self.enable()
        if self._button_pressed_edge(buttons, self.disable_button):
            self.disable()


        self._last_buttons = list(buttons)
        self._last_axes = list(axes)

    # 定时输出：最终 /cmd_vel
    def spin_publish(self):
        if self._latest_joy is None:
            return
        joy = self._latest_joy
        #self.get_logger().info(f"Joy axes: {joy.axes[self.axis_linear]}, buttons: {joy.buttons[self.axis_angular]}")
        # 取轴
        lin = 0.0
        ang = 0.0
        if self.axis_linear < len(joy.axes):
            lin = joy.axes[self.axis_linear]
        if self.axis_angular < len(joy.axes):
            ang = joy.axes[self.axis_angular]
        # 死区
        lin = 0.0 if abs(lin) < self.deadzone else lin
        ang = 0.0 if abs(ang) < self.deadzone else ang
        # 反向
        if self.invert_linear:
            lin = -lin
        if self.invert_angular:
            ang = -ang

        if (not self.power_status):
            self.joy_cmd.linear.x = 0.0
            self.joy_cmd.angular.z = 0.0
        else:
            self.joy_cmd.linear.x = lin * self.scale_linear
            self.joy_cmd.angular.z = ang * self.scale_angular
        # self.get_logger().info(f"Publishing joy cmd_vel: lin={self.joy_cmd.linear.x}, ang={self.joy_cmd.angular.z}")
        self.joy_pub.publish(self.joy_cmd)

        use_nav = (self.nav_button_pressed and self.power_status)
        out = self.nav_cmd if use_nav else self.joy_cmd

        non_zero = (abs(out.linear.x) > 1e-9) or (abs(out.angular.z) > 1e-9)

        if non_zero:
            self.out_pub.publish(out)
            self.last_zero_vel_sent = False
        else:
            if not self.last_zero_vel_sent:
                self.out_pub.publish(out)
                self.last_zero_vel_sent = True

    # MotorPower 控制
    def enable(self):
        self.out_pub.publish(Twist())
        self.last_zero_vel_sent = True

        if not self.power_status:
            msg = MotorPower()
            msg.state = MotorPower.ON
            self.motor_pub.publish(msg)
            self.power_status = True
            self.get_logger().info('Motor power: ON')
        else:
            self.get_logger().warn('Motor power already ON')

    def disable(self):
        self.out_pub.publish(Twist())
        self.last_zero_vel_sent = True

        if self.power_status:
            msg = MotorPower()
            msg.state = MotorPower.OFF
            self.motor_pub.publish(msg)
            self.power_status = False
            self.get_logger().info('Motor power: OFF')
        else:
            self.get_logger().warn('Motor power already OFF')


def main(args=None):
    rclpy.init(args=args)
    node = XBotTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.out_pub.publish(Twist())
            node.disable()  # 不想退出断电就注释
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
