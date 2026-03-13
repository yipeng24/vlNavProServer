#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np


class ThrottleCompress(Node):
    """
    订阅 RealSense 原始图像(30fps)，只以 2Hz 发布压缩后的图像，降低通信压力。
    """
    def __init__(self):
        super().__init__('throttle_compress')
        self.get_logger().info("ThrottleCompress node started.")
        self.color_in = '/realsense2_camera/color/image_raw/compressed'   # 你的实际topic可能是 /camera/color/image_raw
        self.depth_in = '/realsense2_camera/depth/image_rect_raw/compressedDepth'

        self.color_out = '/net/color/image_rgb_compressed_2hz'
        self.depth_out = '/net/depth/image_depth_compressed_2hz'
        self.use_compressed_as_input = True
        self.rate_hz = 2.0
        self.jpeg_quality = 70

        # ============================================

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None

        self.sub_color = self.create_subscription(
            CompressedImage, self.color_in, self.cb_color, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            CompressedImage, self.depth_in, self.cb_depth, qos_profile_sensor_data
        )

        self.pub_color = self.create_publisher(CompressedImage, self.color_out, 10)
        self.pub_depth = self.create_publisher(CompressedImage, self.depth_out, 10)

        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(f"Sub color: {self.color_in} -> Pub: {self.color_out} @ {self.rate_hz}Hz")
        self.get_logger().info(f"Sub depth: {self.depth_in} -> Pub: {self.depth_out} @ {self.rate_hz}Hz")

    def cb_color(self, msg: CompressedImage):
        # self.get_logger().debug(f"Received color image at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        self.latest_color = msg

    def cb_depth(self, msg: CompressedImage):
        self.latest_depth = msg

    def on_timer(self):
        if self.use_compressed_as_input:
            if self.latest_color is not None:
                self.pub_color.publish(self.latest_color)  # 不要再解码编码

            if self.latest_depth is not None:
                self.pub_depth.publish(self.latest_depth)  # 同上


def main():
    rclpy.init()
    node = ThrottleCompress()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
