from dataclasses import dataclass
from enum import Enum, auto
import os
import time
import cv2
import numpy as np
import rclpy
import tf2_ros

import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy,qos_profile_sensor_data

from sensor_msgs.msg import Joy, CameraInfo, CompressedImage
from geometry_msgs.msg import Twist
from kobuki_ros_interfaces.msg import MotorPower
from cv_bridge import CvBridge

from applications.teleop_base import teleop_base
from applications.buffer import ImageRingBuffer
from applications.vlm_api import VLMClient
from applications.image_exta import image_exta

def stamp_to_ns(msg: CompressedImage) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

@dataclass
class nav_info:
    stamp_ns: int
    image_color: np.ndarray
    image_depth: np.ndarray
    image_uv: tuple[int, int]
    local_map_waypoint: tuple[float, float]
    vlm_result: dict


@dataclass
class ILGP_State(Enum):
    INIT = auto(),
    TEST_VLM = auto(),
    WAIT_TRIGGER = auto(),
    INFER_VLM = auto(),
    PLAN_TRAJ = auto(),
    Moving = auto(),


class singleTh_container(Node):
    def __init__(self):
        super().__init__("singleTh_container")
        # plugin-1: movement, initialize teleop_base
        self._teleop_base: teleop_base = teleop_base()
        self._image_pool_ring: ImageRingBuffer = ImageRingBuffer(maxlen=30, sync_tolerance_ms=200)
        self.vlm_client: VLMClient = VLMClient()
        self.image_exta: image_exta = image_exta()
        self._state: ILGP_State = ILGP_State.INIT
        self.result_nav_info: nav_info = None

        self.rgb_topic = '/net/color/image_rgb_compressed_2hz'
        self.depth_topic = '/net/depth/image_depth_compressed_2hz'
        self.camera_info_topic = '/realsense2_camera/color/camera_info'
        self.save_dir = os.path.expanduser('/home/yipeng/image_bridge_saved')

        self.nav_vel = Twist()
        self._sub_joy = self.create_subscription(Joy, "/joy", self.joy_callback, 10)
        self._sub_nav = self.create_subscription(Twist, "/cmd_vel_nav", self.nav_callback, 10)
        self.sub_rgb = self.create_subscription(CompressedImage, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(CompressedImage, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.sub_camera_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data) 
        motor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.joy_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.out_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.motor_pub = self.create_publisher(MotorPower, 'motor_power', motor_qos)
        self.set_motor_power(True)  # power on at start

        self.img4show = None
        self.flip_rgb,self.flip_depth = True, True
        self.flip_code = -1  # -1: both, 0: vertical, 1: horizontal
        self.bridge = CvBridge()
        self.image_viewer_scale = 1.0
        cv2.namedWindow("ILGP Viewer", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Viewer Current", cv2.WINDOW_NORMAL)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.vlm_idle = True
        self.result_img = None
        self._teleop_timer = self.create_timer(0.1, self._teleop_timer_callback)
        self._image_viewer_timer = self.create_timer(0.1, self._image_viewer_timer_callback)
        

        self._exec_thread = threading.Thread(target=self._exec_ilgp_process, daemon=True)
        self._vlm_thread = threading.Thread(target=self._exec_vlm_process, daemon=True)

        cv2.namedWindow("VLM Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VLM Debug", 640, 480)
        # self._exec_thread.start()
        self._vlm_thread.start()
    
    
    def _teleop_timer_callback(self):
        self.publish_vel()
        if self._teleop_base._trans_snapshot_triggered:
            # trigger snapshot
            if self._image_pool_ring is None:
                self.get_logger().warn("Image pool ring is None")
                return
            if self._image_pool_ring.size() == 0:
                self.get_logger().warn("Image pool ring is empty")
                return
            
            res = self._image_pool_ring.save_latest(self.save_dir)  # save latest image
            self.get_logger().info(f"Snapshot saved status: {res}")
            self._teleop_base._trans_snapshot_triggered = False

        if self._teleop_base._trans_vlm_triggered:
            self.get_logger().info("VLM inference triggered")
            # trigger VLM call
            self._state = ILGP_State.TEST_VLM
            self._teleop_base._trans_vlm_triggered = False


    #展示当前图像
    def _image_viewer_timer_callback(self):
        if self.img4show is None:
            return

        img_bgr = self.img4show 
        
        vis = img_bgr.copy()
        cv2.putText(vis, f"{self._state}  |   vlnidle: {self.vlm_idle}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.imshow("ILGP Viewer", vis)
        if self.result_img is not None:
            cv2.imshow("VLM Debug", self.result_img)
        # else:
        #     cv2.imshow("VLM Debug", img_bgr)
        cv2.waitKey(1)

        # pack = self._image_pool_ring.get_latest(1)
        # if pack is None or len(pack) == 0:
        #     self.get_logger().warn("No image pack available for viewing")
        #     cv2.waitKey(1)
        #     return

        # cv2.putText(vis, f"stamp(ns): {pack[0].stamp_ns}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # cv2.putText(vis, f"ring: {self._image_pool_ring.size()}/{self._image_pool_ring.maxlen}", (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


    def joy_callback(self, msg: Joy):
        self._teleop_base.joy_update(msg)


    def nav_callback(self, msg: Twist):
        self.nav_vel = msg


    def camera_info_callback(self, msg: CameraInfo):
        self.image_exta.update_camera_info(msg)


    def rgb_callback(self, msg: CompressedImage):
        try:
            self.img4show = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"RGB cv_bridge failed: {e}")
            return
        if self.flip_rgb:
            self.img4show = cv2.flip(self.img4show, self.flip_code)
        self._image_pool_ring.update_rgb(stamp_to_ns(msg), self.img4show)


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

        self._image_pool_ring.update_depth(stamp_to_ns(msg), depth)


    def save_latest_to_disk(self, n: int = 1):
        packs = self._image_pool_ring.get_latest(n)
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
        power_msg.state = MotorPower.ON if enable else MotorPower.OFF
        self.motor_pub.publish(power_msg)


    def _exec_vlm_process(self):
        self.vlm_idle = True
        while rclpy.ok():
            if self.vlm_idle:
                if self._state is ILGP_State.TEST_VLM:
                    #1. 从图像池获取最近的4帧图像（如果有的话）
                    packs = []
                    if self._image_pool_ring.size() == 0:
                        continue
                    elif self._image_pool_ring.size() < 4:
                        packs = self._image_pool_ring.get_latest(self._image_pool_ring.size())
                    else:
                        packs = self._image_pool_ring.get_latest(4)
                    #2. 将图像送入VLM API进行推理
                    self.vlm_idle = False
                    if not packs:
                        self.get_logger().warn("No images for VLM")
                        continue

                    rgb_list = [p.rgb_bgr.copy() for p in packs]

                    instruction =  "go to the door" # input("input VLM instruction: ")
                    self.get_logger().info(f"Sending VLM instruction: {instruction}")

                    res_raw = self.vlm_client.infer_vlm(instruction, rgb_list)
                    self.get_logger().info(f"VLM infer result: {res_raw}")

                    if res_raw['ok'] is not True:
                        continue

                    self.result_nav_info = nav_info(
                        stamp_ns = packs[-1].stamp_ns,
                        image_color = packs[-1].rgb_bgr,
                        image_depth = packs[-1].depth,
                        image_uv = res_raw.get('uv', (-1, -1)),
                        vlm_result = res_raw,
                        local_map_waypoint = None
                    )

                    self.result_img = rgb_list[-1].copy()
                    cv2.circle(self.result_img, (res_raw['uv'][0], res_raw['uv'][1]), 10, (0,255,0), 2)
                    
                    self.vlm_idle = True
                    self._state = ILGP_State.WAIT_TRIGGER

                    try:
                        T_map_from_cam = self.tf_buffer.lookup_transform("odom", "camera_link", rclpy.time.Time())
                    except tf2_ros.LookupException:
                        self.get_logger().warn("TF lookup failed for odom->camera_link")
                        T_map_from_cam = None
                        self.get_logger().warn("Skipping point computation due to missing TF")
                        continue
                    self.image_exta.process_uv_to_point(
                        self.result_nav_info.image_uv,
                        self.result_nav_info.image_depth,
                        T_map_from_cam
                    )
                elif self._state is ILGP_State.INFER_VLM:
                    self._state = ILGP_State.WAIT_TRIGGER
                    continue


            time.sleep(0.1)


    def _exec_ilgp_process(self):
        while rclpy.ok():
            if self._state == ILGP_State.WAIT_TRIGGER:
                pass  # TODO: wait for trigger
            elif self._state == ILGP_State.PLAN_TRAJ:
                pass  # TODO: plan trajectory
            elif self._state == ILGP_State.Moving:
                pass  # TODO: execute movement
            time.sleep(0.1)


def main():
    rclpy.init()
    node = singleTh_container()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
