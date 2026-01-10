#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from kobuki_ros_interfaces.msg import MotorPower
# from image_bridge.image_bridge_node import ImageBridgeNode
from interfaces.srv import SaveLatest
@dataclass
class FramePack:
    stamp_ns: int
    rgb_bgr: Optional[np.ndarray]   # HxWx3 uint8
    depth_u16: Optional[np.ndarray] # HxW uint16 (often mm)


class XBotTeleop(Node):
    def __init__(self):
        super().__init__('xbot_teleop')
        # self.img_bridge = ImageBridgeNode()
        # =====topic config=====
        self.joy_topic = '/joy'
        self.cmd_vel_joy_topic = '/cmd_vel_joy'
        self.cmd_vel_nav_topic = '/cmd_vel_nav'
        self.cmd_vel_out_topic = '/cmd_vel'
        self.motor_power_topic = 'motor_power'  # kobuki_node 通常就是这个
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

        self.nav_hold_button = 0     # A 键
        self.require_nav_hold = True # True=按住A才用Nav2速度（更安全）
        self.deadman_button = -1     

        self.joy_cmd = Twist()       # 手柄算出来的速度
        self.nav_cmd = Twist()       # 订阅到的 Nav2 速度
        self.power_status = False
        self.last_zero_vel_sent = True

        self._last_buttons = None
        self._last_axes = None
        self.nav_button_pressed = False

        self.enable_snapshot = True
        self.snapshot_button = 2          # 默认 X=2（不对就改）
        self.save_depth_png = True
        self.save_depth_npy = False       # 不需要就 False

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

        # Timer: 定时发布
        period = 1.0 / max(1e-6, float(self.publish_rate_hz))
        self.timer = self.create_timer(period, self.spin_publish)

        self.save_cli = self.create_client(SaveLatest, '/save_latest')
        if self.enable_on_start:
            self.enable()

        self.get_logger().info(
            f'XBot teleop+mux started.\n'
            f'  joy={self.joy_topic}\n'
            f'  joy_cmd={self.cmd_vel_joy_topic}\n'
            f'  nav_cmd={self.cmd_vel_nav_topic}\n'
            f'  OUT={self.cmd_vel_out_topic}\n'
            f'  A(button[{self.nav_hold_button}]) hold => use Nav2 speed\n'
        )
        if self.enable_snapshot:
            self.get_logger().info(
                f'📸 Snapshot enabled.\n'
                f'  press button[{self.snapshot_button}] to save current\n'
            )

    # -------------------------
    # 工具函数
    # -------------------------
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

    def _deadman_pressed(self, buttons) -> bool:
        # 默认 deadman_button=-1 => 永远允许（不改变你现有行为）
        if self.deadman_button < 0:
            return True
        if self.deadman_button >= len(buttons):
            return False
        return buttons[self.deadman_button] == 1
    
    def trigger_save(self):
        req = SaveLatest.Request()
        req.n = 1
        req.out_dir = "/home/yipeng/image_bridge_saved"
        req.save_depth_png = True
        req.save_depth_npy = False
        fut = self.save_cli.call_async(req)
        self.get_logger().info(f"📸 Saved current frame to: {req.out_dir}")

    def save_current_frame(self):
        # self.img_bridge.save_latest_to_disk(n=1)
        self.trigger_save()



    def nav_callback(self, msg: Twist):
        self.nav_cmd = msg


    def joy_callback(self, msg: Joy):
        axes = msg.axes
        buttons = msg.buttons

        # A键状态（按住用Nav2）
        self.nav_button_pressed = self._nav_hold_pressed(buttons)

        if self.enable_snapshot and self._button_pressed_edge(buttons, self.snapshot_button):
            self.save_current_frame()

        # enable/disable：按下沿触发
        if self._button_pressed_edge(buttons, self.enable_button):
            self.enable()
        if self._button_pressed_edge(buttons, self.disable_button):
            self.disable()

        # 读轴
        try:
            raw_lin = float(axes[self.axis_linear])
            raw_ang = float(axes[self.axis_angular])
        except Exception:
            self.get_logger().warn('Joy axes index out of range. Check axis_linear/axis_angular.')
            self._last_buttons = list(buttons)
            self._last_axes = list(axes)
            return

        lin = self._apply_deadzone(raw_lin, self.deadzone)
        ang = self._apply_deadzone(raw_ang, self.deadzone)

        if self.invert_linear:
            lin = -lin
        if self.invert_angular:
            ang = -ang

        # deadman 或未上电：手柄速度强制0
        if (not self._deadman_pressed(buttons)) or (not self.power_status):
            self.joy_cmd.linear.x = 0.0
            self.joy_cmd.angular.z = 0.0
        else:
            self.joy_cmd.linear.x = lin * self.scale_linear
            self.joy_cmd.angular.z = ang * self.scale_angular

        # 发布一份 /cmd_vel_joy 方便你调试
        self.joy_pub.publish(self.joy_cmd)

        self._last_buttons = list(buttons)
        self._last_axes = list(axes)

    # -------------------------
    # 定时输出：最终 /cmd_vel
    # -------------------------
    def spin_publish(self):
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

    # -------------------------
    # MotorPower 控制
    # -------------------------
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
