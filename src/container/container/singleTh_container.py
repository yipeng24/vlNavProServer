from dataclasses import dataclass
from enum import Enum, auto
import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy,qos_profile_sensor_data

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from kobuki_ros_interfaces.msg import MotorPower
from cv_bridge import CvBridge

from applications.teleop_base import teleop_base
from applications.buffer import ImageRingBuffer
def stamp_to_ns(msg: CompressedImage) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


@dataclass
class ILGP_State(Enum):
    INIT = auto(),
    WAIT_TRIGGER = auto(),
    PLAN_TRAJ = auto(),
    Moving = auto(),


class singleTh_container(Node):
    def __init__(self):
        super().__init__("singleTh_container")
        # plugin-1: movement, initialize teleop_base
        self._teleop_base: teleop_base = teleop_base()
        self._image_pool_ring: ImageRingBuffer = ImageRingBuffer(maxlen=30, sync_tolerance_ms=200)
        self._state: ILGP_State = ILGP_State.INIT

        self.nav_vel = Twist()
        self._sub_joy = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 10)
        self._sub_nav = self.create_subscription(Twist, self.cmd_vel_nav_topic, self.nav_callback, 10)

        motor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.joy_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.out_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.motor_pub = self.create_publisher(MotorPower, 'motor_power', motor_qos)
        self.set_motor_power(True)  # power on at start

        self.rgb_topic = '/net/color/image_rgb_compressed_2hz'
        self.depth_topic = '/net/depth/image_depth_compressed_2hz'
        self.save_dir = os.path.expanduser('/home/yipeng/image_bridge_saved')
        self.flip_rgb,self.flip_depth = True, True
        self.flip_code = -1  # -1: both, 0: vertical, 1: horizontal
        self.bridge = CvBridge()
        self.image_viewer_scale = 1.0
        cv2.namedWindow("ILGP Viewer", cv2.WINDOW_NORMAL)
        self.sub_rgb = self.create_subscription(CompressedImage, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(CompressedImage, self.depth_topic, self.depth_callback, qos_profile_sensor_data)

        self._teleop_timer = self.create_timer(0.1, self._teleop_timer_callback)
        
    
    def _teleop_timer_callback(self):
        self.publish_vel()
        if self._teleop_base._trans_snapshot_trigger:
            #TODO: trigger snapshot
            if self._image_pool_ring is None:
                self.get_logger().warn("Image pool ring is None")
                return
            if self._image_pool_ring.size() == 0:
                self.get_logger().warn("Image pool ring is empty")
                return
            
            res = self._image_pool_ring.save_latest(self.save_dir)  # save latest image
            self.get_logger().info(f"Snapshot saved status: {res}")
            self._teleop_base._trans_snapshot_trigger = False
            
    def _image_viewer_timer_callback(self):
        pack = self._image_pool_ring.get_latest(1)
        if pack is None or len(pack) == 0:
            self.get_logger().warn("No image pack available for viewing")
            cv2.waitKey(1)
            return
        img_bgr = pack[0].rgb_bgr
        vis = img_bgr.copy()
        cv2.putText(vis, f"stamp(ns): {pack[0].stamp_ns}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(vis, f"ring: {self._image_pool_ring.size()}/{self._image_pool_ring.maxlen}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.imshow("ILGP Viewer", vis)
        cv2.waitKey(1)
    def joy_callback(self, msg: Joy):
        self._teleop_base.joy_update(msg)


    def nav_callback(self, msg: Twist):
        self.nav_vel = msg


    def rgb_callback(self, msg: CompressedImage):
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"RGB cv_bridge failed: {e}")
            return
        if self.flip_rgb:
            bgr = cv2.flip(bgr, self.flip_code)
        self.ring.update_rgb(stamp_to_ns(msg), bgr)


    def depth_callback(self, msg: CompressedImage):
        try:
            if "compressedDepth" in msg.format:
                raw = bytes(msg.data)
                png_sig = b"\x89PNG\r\n\x1a\n"

                idx = raw.find(png_sig)
                if idx < 0:
                    raise RuntimeError(f"PNG signature not found. first16={raw[:16]!r} len={len(raw)}")

                png_bytes = raw[idx:]
                png = np.frombuffer(png_bytes, dtype=np.uint8)
                depth = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    raise RuntimeError(f"cv2.imdecode returned None even after PNG slice. idx={idx}, first8={png_bytes[:8]!r}")
            else:
                depth = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

        except Exception as e:
            self.get_logger().warn(f"Depth decode failed: {e}")
            return

        if self.flip_depth:
            depth = cv2.flip(depth, self.flip_code)

        self.ring.update_depth(stamp_to_ns(msg), depth)


    def save_latest_to_disk(self, n: int = 1):
        packs = self.ring.get_latest(n)
        jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        for p in packs:
            prefix = os.path.join(self.save_dir, f"{p.stamp_ns}")
            rgb_path = f"{prefix}_rgb.jpg"
            cv2.imwrite(rgb_path, p.rgb_bgr, jpg_params)

        self.get_logger().info(f"Saved {len(packs)} packs to {self.save_dir}")


    def publish_vel(self):
        joy_vel = Twist()
        if self._teleop_base._trans_use_nav:
            joy_vel = self.nav_vel
        else:
            joy_vel.linear.x, joy_vel.angular.z = self._teleop_base.calc_cmd_vel()

        self.joy_pub.publish(joy_vel)
        

    def set_motor_power(self, enable: bool):
        power_msg = MotorPower()
        power_msg.state = MotorPower.STATE_ON if enable else MotorPower.STATE_OFF
        self.motor_pub.publish(power_msg)

    