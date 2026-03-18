from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import time
from std_msgs.msg import String
import cv2
import numpy as np
import rclpy
import tf2_ros

import threading

from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy,qos_profile_sensor_data

from sensor_msgs.msg import Joy, CameraInfo, CompressedImage
from geometry_msgs.msg import Twist,Pose2D
from kobuki_ros_interfaces.msg import MotorPower
from cv_bridge import CvBridge

from applications.teleop_base import teleop_base
from applications.buffer import ImageRingBuffer
# from applications.vlm_api import VLMClient
from applications.vlm_openai_api import VLMClient
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
    WAIT_TRIGGER = auto(), # 中间挂起状态
    INFER_VLM = auto(),    # 进行VLM推理
    Moving = auto(),       

USER_INSTRUCTION = "go straight and turn right, then go straight to the white trash can and stop there"
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
        self.pending_goal_pose: tuple[float, float, float] | None = None
        self.vlm_result_ready = False
        self._vlm_request_pending = False
        self.nav_goal_dispatched = False
        self._moving_wait_logged = False
        self._infer_wait_logged = False
        self._last_logged_state: ILGP_State | None = None
        self._last_nav_takeover_logged: bool | None = None

        self.rgb_topic = '/net/color/image_rgb_compressed_2hz'
        self.depth_topic = '/net/depth/image_depth_compressed_2hz'
        self.camera_info_topic = '/realsense2_camera/color/camera_info'
        self.save_dir = os.path.expanduser('/home/yipeng/image_bridge_saved')
        os.makedirs(self.save_dir, exist_ok=True)
        self.vlm_raw_text_log_path = os.path.join(self.save_dir, 'vlm_raw_text.log')

        self.bridge_goal_topic = '/top/goal_pose2d'
        self.bridge_status_topic = '/top/nav_status'
        self.bridge_feedback_topic = '/top/nav_feedback'
        self.bridge_result_topic = '/top/nav_result'

        self.latest_nav_status = ''
        self.latest_nav_feedback = ''
        self.latest_nav_result = ''
        self._last_logged_nav_status: str | None = None
        self.last_sent_goal = None
        self._last_nav_vel_rx_monotonic = 0.0

        self.nav_vel = Twist()
        self._sub_joy = self.create_subscription(Joy, "/joy", self.joy_callback, 10)
        self._sub_nav = self.create_subscription(Twist, "/cmd_vel_nav", self.nav_callback, 10)
        self.sub_rgb = self.create_subscription(CompressedImage, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(CompressedImage, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.sub_camera_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data) 
        
        self.bridge_goal_pub = self.create_publisher(Pose2D, self.bridge_goal_topic, 10)

        self.bridge_status_sub = self.create_subscription(
            String, self.bridge_status_topic, self.bridge_status_callback, 10
        )
        self.bridge_feedback_sub = self.create_subscription(
            String, self.bridge_feedback_topic, self.bridge_feedback_callback, 10
        )
        self.bridge_result_sub = self.create_subscription(
            String, self.bridge_result_topic, self.bridge_result_callback, 10
        )

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
        self.latest_rgb_for_vlm = None
        self.latest_rgb_stamp_ns = None
        self.flip_rgb,self.flip_depth = True, True
        self.flip_code = -1  # -1: both, 0: vertical, 1: horizontal
        self.bridge = CvBridge()
        self.image_viewer_scale = 1.0
        cv2.namedWindow("ILGP Viewer", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Viewer Current", cv2.WINDOW_NORMAL)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.declare_parameter('goal_frame', 'map')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('base_frame', 'base_footprint')
        self.goal_frame = str(self.get_parameter('goal_frame').value)
        configured_camera_frame = str(self.get_parameter('camera_frame').value).strip()
        self.camera_frame = configured_camera_frame if configured_camera_frame else None
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.get_logger().info(
            f"Goal TF config: goal_frame={self.goal_frame}, "
            f"camera_frame={self.camera_frame or '<from CameraInfo>'}, "
            f"base_frame={self.base_frame}"
        )

        self.vlm_idle = True
        self.ensure_nav2 = False
        self.result_img = None
        self._teleop_timer = self.create_timer(0.1, self._teleop_timer_callback)
        self._image_viewer_timer = self.create_timer(0.1, self._image_viewer_timer_callback)
        

        self._exec_thread = threading.Thread(target=self._exec_ilgp_process, daemon=True)
        self._vlm_thread = threading.Thread(target=self._exec_vlm_process, daemon=True)

        cv2.namedWindow("VLM Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VLM Debug", 640, 480)
        self._exec_thread.start()
        self._vlm_thread.start()
    
    def _queue_vlm_inference(self):
        self.result_nav_info = None
        self.pending_goal_pose = None
        self.vlm_result_ready = False
        self.nav_goal_dispatched = False
        self._moving_wait_logged = False
        self._infer_wait_logged = False
        self.latest_nav_status = ''
        self.latest_nav_feedback = ''
        self.latest_nav_result = ''
        self._vlm_request_pending = True
        self._state = ILGP_State.INFER_VLM


    def _log_vlm_raw_text(self, res_raw: dict):
        raw_text = ""
        sta_text = ""
        error_text = ""
        if isinstance(res_raw, dict):
            raw_text = str(res_raw.get("raw_text", "") or "").strip()
            sta_text = str(res_raw.get("sta", "") or "").strip()
            error_text = str(res_raw.get("error", "") or "").strip()

        header_parts = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
        if sta_text:
            header_parts.append(f"sta={sta_text}")
        if error_text:
            header_parts.append(f"error={error_text}")

        text_to_store = raw_text if raw_text else "<empty>"
        try:
            with open(self.vlm_raw_text_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{' | '.join(header_parts)}]\n{text_to_store}\n\n")
        except OSError as e:
            self.get_logger().warn(f"Failed to write VLM raw_text log to {self.vlm_raw_text_log_path}: {e}")

        if raw_text:
            self.get_logger().info(f"[VLM raw_text]\n{raw_text}")
            return

        if error_text:
            self.get_logger().info(f"[VLM raw_text] <empty> error={error_text}")
            return

        self.get_logger().info("[VLM raw_text] <empty>")

    
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
            if self.result_img is not None:
                snapshot_path = os.path.join(self.save_dir, f"vlm_result_{self._image_pool_ring.size()}.jpg")
                cv2.imwrite(snapshot_path, self.result_img)
                self.get_logger().info(f"Snapshot result image saved to {snapshot_path}")

        if self._teleop_base._trans_vlm_triggered:
            self.get_logger().info("VLM inference triggered")
            self._queue_vlm_inference()
            self._teleop_base._trans_vlm_triggered = False

        if self._teleop_base._ensure_nav_triggered:
            self.get_logger().info("Ensure Nav2 triggered by teleop")
            self.ensure_nav2 = True 
            self._teleop_base._ensure_nav_triggered = False


    #展示当前图像
    def _image_viewer_timer_callback(self):
        if self.img4show is None:
            return

        img_bgr = self.img4show
        vis = img_bgr.copy()

        if self._image_pool_ring is not None:
            cv2.putText(vis, f"ring: {self._image_pool_ring.size()}/{self._image_pool_ring.maxlen}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 

        cv2.putText(vis, f"{self._state}  |   vlnIdle: {self.vlm_idle}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(vis, f"nav_status: {self.latest_nav_status}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.putText(vis, f"nav_result: {self.latest_nav_result}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Build a single-row history preview (4 frames) and merge into ILGP Viewer.
        h, w = vis.shape[:2]
        thumb_h = max(80, h // 5)
        thumb_w = max(120, w // 4)

        history_frames = []
        if self._image_pool_ring is not None and self._image_pool_ring.size() > 0:
            history_frames = [p.rgb_bgr for p in self._image_pool_ring.get_latest(4)]

        # Keep left->right as old->new, pad with empty images on the left when missing.
        padded_frames = [None] * (4 - len(history_frames)) + history_frames

        thumbs = []
        for frame in padded_frames:
            if frame is None:
                thumb = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            else:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb)

        history_strip = np.hstack(thumbs)
        history_strip = cv2.resize(history_strip, (w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.putText(history_strip, "history x4 (old -> new)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        merged = np.vstack([vis, history_strip])
        cv2.imshow("ILGP Viewer", merged)
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
        if self._teleop_base._trans_use_nav != self._last_nav_takeover_logged:
            mode = "Nav2 /cmd_vel_nav passthrough" if self._teleop_base._trans_use_nav else "manual joystick cmd_vel"
            self.get_logger().info(
                f"Velocity control mode -> {mode}; buttons={list(msg.buttons)} axes={list(msg.axes)}"
            )
            self._last_nav_takeover_logged = self._teleop_base._trans_use_nav
            if self._teleop_base._trans_use_nav:
                self.joy_pub.publish(self.nav_vel)
        if self._teleop_base._call_vlm_button_pressed:
            self.get_logger().info(
                f"LB edge detected: buttons={list(msg.buttons)} axes={list(msg.axes)}"
            )
        if self._teleop_base._ensure_nav_button_pressed:
            self.get_logger().info("Y edge detected: ensure_nav2 requested")
        if self._teleop_base.snapshot_button_pressed:
            self.get_logger().info("B edge detected: snapshot requested")


    def nav_callback(self, msg: Twist):
        self.nav_vel = Twist()
        self.nav_vel.linear.x = msg.linear.x
        self.nav_vel.linear.y = msg.linear.y
        self.nav_vel.linear.z = msg.linear.z
        self.nav_vel.angular.x = msg.angular.x
        self.nav_vel.angular.y = msg.angular.y
        self.nav_vel.angular.z = msg.angular.z
        self._last_nav_vel_rx_monotonic = time.monotonic()

        if self._teleop_base._trans_use_nav:
            self.joy_pub.publish(self.nav_vel)


    def camera_info_callback(self, msg: CameraInfo):
        self.image_exta.update_camera_info(msg)
        if self.camera_frame is None and msg.header.frame_id:
            self.camera_frame = msg.header.frame_id
            self.get_logger().info(
                f"Using camera frame from CameraInfo: {self.camera_frame}"
            )


    def rgb_callback(self, msg: CompressedImage):
        try:
            self.img4show = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"RGB cv_bridge failed: {e}")
            return
        if self.flip_rgb:
            self.img4show = cv2.flip(self.img4show, self.flip_code)

        # Cache latest RGB, and only push to ring when VLM is triggered.
        self.latest_rgb_for_vlm = self.img4show.copy()
        self.latest_rgb_stamp_ns = stamp_to_ns(msg)


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


    def bridge_status_callback(self, msg: String):
        self.latest_nav_status = msg.data
        if msg.data != self._last_logged_nav_status:
            self.get_logger().info(f"[bridge status] {msg.data}")
            self._last_logged_nav_status = msg.data


    def bridge_feedback_callback(self, msg: String):
        self.latest_nav_feedback = msg.data
        self.get_logger().info(f"[bridge feedback] {msg.data}")


    def bridge_result_callback(self, msg: String):
        self.latest_nav_result = msg.data
        self.get_logger().info(f"[bridge result] {msg.data}")


    def publish_goal_to_bridge(self, x: float, y: float, yaw: float):
        msg = Pose2D()
        msg.x = float(x)
        msg.y = float(y)
        msg.theta = float(yaw)

        self.bridge_goal_pub.publish(msg)
        self.last_sent_goal = (msg.x, msg.y, msg.theta)

        self.get_logger().info(
            f"[bridge goal pub] x={msg.x:.3f}, y={msg.y:.3f}, yaw={msg.theta:.3f}"
        )


    def _exec_vlm_process(self):
        self.vlm_idle = True
        while rclpy.ok():
            if not self.vlm_idle or self._state is not ILGP_State.INFER_VLM or not self._vlm_request_pending:
                time.sleep(0.1)
                continue

            self._vlm_request_pending = False
            self.vlm_idle = False
            try:
                if self.latest_rgb_for_vlm is None or self.latest_rgb_stamp_ns is None:
                    self.get_logger().warn("No cached RGB available for VLM trigger")
                    self._state = ILGP_State.WAIT_TRIGGER
                    continue

                pre_pair_count = self._image_pool_ring.size()

                # Push RGB into ring only when a fresh VLM request is queued.
                self._image_pool_ring.update_rgb(
                    self.latest_rgb_stamp_ns,
                    self.latest_rgb_for_vlm
                )

                # Give the next depth callback a short window to pair with the queued RGB.
                wait_deadline = time.monotonic() + 0.70
                while self._image_pool_ring.size() <= pre_pair_count and time.monotonic() < wait_deadline:
                    time.sleep(0.02)

                #1. 从图像池获取最近的图像
                packs = []
                if self._image_pool_ring.size() <= pre_pair_count:
                    sync_info = self._image_pool_ring.pending_sync_info()
                    dt_ms = None
                    if sync_info["dt_ns"] is not None:
                        dt_ms = sync_info["dt_ns"] / 1e6
                    self.get_logger().warn(
                        "No synced RGB/Depth pack after VLM-triggered RGB push; "
                        f"rgb_ns={sync_info['rgb_ns']} depth_ns={sync_info['depth_ns']} "
                        f"dt_ms={dt_ms} tol_ms={sync_info['tol_ns'] / 1e6:.1f}"
                    )
                    self._state = ILGP_State.WAIT_TRIGGER
                    continue
                if self._image_pool_ring.size() >= 1:
                    packs = self._image_pool_ring.get_latest(1)
                else:
                    packs = self._image_pool_ring.get_latest(self._image_pool_ring.size())

                #2. 将图像送入VLM API进行推理
                if not packs:
                    self.get_logger().warn("No images for VLM")
                    self.result_nav_info = None
                    self.pending_goal_pose = None
                    self.vlm_result_ready = True
                    continue

                rgb_list = [p.rgb_bgr.copy() for p in packs]

                instruction =  USER_INSTRUCTION # input("input VLM instruction: ")
                self.get_logger().info(f"Sending VLM instruction: {instruction}")

                res_raw = self.vlm_client.infer_vlm(instruction, rgb_list)
                self._log_vlm_raw_text(res_raw)
                self.get_logger().info(f"VLM infer result: {res_raw}")

                if res_raw['ok'] is not True:
                    self.result_nav_info = None
                    self.pending_goal_pose = None
                    self.vlm_result_ready = True
                    continue

                # 3. 将VLM结果、图像和深度等信息打包，并计算目标点在地图坐标系中的位置
                self.result_nav_info = nav_info(
                    stamp_ns = packs[-1].stamp_ns,
                    image_color = packs[-1].rgb_bgr,
                    image_depth = packs[-1].depth,
                    image_uv = res_raw.get('uv', (-1, -1)),
                    vlm_result = res_raw,
                    local_map_waypoint = None
                )
                
                self.result_img = rgb_list[-1].copy()
                waypoint_uv = self.result_nav_info.image_uv
                img_h, img_w = self.result_img.shape[:2]
                uv_in_bounds = (
                    waypoint_uv is not None and
                    len(waypoint_uv) == 2 and
                    0 <= int(waypoint_uv[0]) < img_w and
                    0 <= int(waypoint_uv[1]) < img_h
                )
                if uv_in_bounds:
                    cv2.circle(self.result_img, (int(waypoint_uv[0]), int(waypoint_uv[1])), 10, (0,255,0), 2)

                waypoint_depth_m = self.image_exta.get_depth_meters(
                    waypoint_uv,
                    self.result_nav_info.image_depth
                )
                if waypoint_depth_m is not None:
                    depth_text = f"waypoint depth: {waypoint_depth_m:.3f} m"
                    self.get_logger().info(
                        f"Waypoint depth at uv={tuple(map(int, waypoint_uv))}: {waypoint_depth_m:.3f} m"
                    )
                    depth_color = (0, 255, 0)
                else:
                    depth_text = "waypoint depth: N/A"
                    self.get_logger().warn(f"Waypoint depth unavailable at uv={waypoint_uv}")
                    depth_color = (0, 0, 255)

                # cv2.putText(
                #     self.result_img,
                #     depth_text,
                #     (10, 30),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.8,
                #     depth_color,
                #     2
                # )

                if self.camera_frame is None:
                    self.get_logger().warn("Skipping point computation because camera_frame is not available yet")
                    self.pending_goal_pose = None
                    self.vlm_result_ready = True
                    continue
                try:
                    T_goal_from_cam = self.tf_buffer.lookup_transform(
                        self.goal_frame,
                        self.camera_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.2)
                    )
                except tf2_ros.TransformException as e:
                    self.get_logger().warn(
                        f"TF lookup failed for {self.goal_frame}->{self.camera_frame}: {e}"
                    )
                    self.pending_goal_pose = None
                    self.vlm_result_ready = True
                    self.get_logger().warn("Skipping point computation due to missing TF")
                    continue
                # 4. 将计算得到的目标点发送给Bridge节点
                pt_map = self.image_exta.process_uv_to_point(
                    self.result_nav_info.image_uv,
                    self.result_nav_info.image_depth,
                    T_goal_from_cam
                )

                if pt_map is None:
                    self.get_logger().warn("process_uv_to_point returned None")
                    self.pending_goal_pose = None
                    self.vlm_result_ready = True
                    continue

                # 兼容 (x, y) / (x, y, z)
                goal_x = float(pt_map[0])
                goal_y = float(pt_map[1])

                # 在目标坐标系中，让机器人朝向目标点
                try:
                    T_goal_from_base = self.tf_buffer.lookup_transform(
                        self.goal_frame,
                        self.base_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.2)
                    )
                    base_x = float(T_goal_from_base.transform.translation.x)
                    base_y = float(T_goal_from_base.transform.translation.y)
                    yaw = math.atan2(goal_y - base_y, goal_x - base_x)
                except tf2_ros.TransformException as e:
                    self.get_logger().warn(
                        f"TF lookup failed for {self.goal_frame}->{self.base_frame}, "
                        f"fallback yaw uses origin. detail: {e}"
                    )
                    yaw = math.atan2(goal_y, goal_x)

                self.result_nav_info.local_map_waypoint = (goal_x, goal_y)
                self.pending_goal_pose = (goal_x, goal_y, yaw)
                self.vlm_result_ready = True
            finally:
                self.vlm_idle = True

            time.sleep(0.1)


    def _exec_ilgp_process(self):
        while rclpy.ok():
            if self._state is not self._last_logged_state:
                self.get_logger().info(f"ILGP state -> {self._state.name}")
                self._last_logged_state = self._state

            if self._state is ILGP_State.INIT:
                self.get_logger().info("ILGP state machine initialized -> WAIT_TRIGGER")
                self._state = ILGP_State.WAIT_TRIGGER

            elif self._state is ILGP_State.WAIT_TRIGGER:
                pass

            elif self._state is ILGP_State.INFER_VLM:
                if not self.vlm_result_ready and self.vlm_idle and not self._infer_wait_logged:
                    self.get_logger().info("INFER_VLM active: waiting for VLM worker result")
                    self._infer_wait_logged = True
                if not self.vlm_result_ready:
                    pass
                elif self.result_nav_info is None:
                    self.get_logger().warn("VLM finished without a valid result, back to WAIT_TRIGGER")
                    self._infer_wait_logged = False
                    self.vlm_result_ready = False
                    self.pending_goal_pose = None
                    self._state = ILGP_State.WAIT_TRIGGER
                else:
                    self._infer_wait_logged = False
                    sta = str(self.result_nav_info.vlm_result.get('sta', '')).strip().lower()
                    if sta == 'finish':
                        self.get_logger().info("VLM reports instruction finished, back to WAIT_TRIGGER")
                        self.vlm_result_ready = False
                        self.pending_goal_pose = None
                        self.ensure_nav2 = False
                        self._state = ILGP_State.WAIT_TRIGGER
                    elif sta == 'move':
                        if self.pending_goal_pose is None:
                            self.get_logger().warn("VLM requested move but no waypoint was computed")
                            self.vlm_result_ready = False
                            self._state = ILGP_State.WAIT_TRIGGER
                        else:
                            goal_x, goal_y, goal_yaw = self.pending_goal_pose
                            self.get_logger().info(
                                f"VLM produced waypoint ({goal_x:.3f}, {goal_y:.3f}, {goal_yaw:.3f}), "
                                "switching to Moving"
                            )
                            self.vlm_result_ready = False
                            self.nav_goal_dispatched = False
                            self._moving_wait_logged = False
                            self._state = ILGP_State.Moving
                    elif sta == 'noway':
                        self.get_logger().warn("VLM reports instruction is currently impossible, back to WAIT_TRIGGER")
                        self.vlm_result_ready = False
                        self.pending_goal_pose = None
                        self.ensure_nav2 = False
                        self._state = ILGP_State.WAIT_TRIGGER
                    else:
                        self.get_logger().warn(f"Unknown VLM state '{sta}', back to WAIT_TRIGGER")
                        self.vlm_result_ready = False
                        self.pending_goal_pose = None
                        self.ensure_nav2 = False
                        self._state = ILGP_State.WAIT_TRIGGER

            elif self._state is ILGP_State.Moving:
                if self.pending_goal_pose is None:
                    self.get_logger().warn("Moving state has no pending goal, back to WAIT_TRIGGER")
                    self.nav_goal_dispatched = False
                    self._moving_wait_logged = False
                    self.ensure_nav2 = False
                    self._state = ILGP_State.WAIT_TRIGGER
                elif not self.ensure_nav2:
                    if not self._moving_wait_logged:
                        self.get_logger().info("Moving state is waiting for ensure_nav2=True before dispatching Nav2 goal")
                        self._moving_wait_logged = True
                else:
                    if not self.nav_goal_dispatched:
                        goal_x, goal_y, goal_yaw = self.pending_goal_pose
                        self.publish_goal_to_bridge(goal_x, goal_y, goal_yaw)
                        self.nav_goal_dispatched = True
                        self._moving_wait_logged = False
                        self.get_logger().info("Nav2 goal dispatched, waiting for navigation result")
                    elif self.latest_nav_result == 'success':
                        self.get_logger().info("Navigation succeeded, back to WAIT_TRIGGER for the next LB trigger")
                        self.latest_nav_result = ''
                        self.latest_nav_status = ''
                        self.latest_nav_feedback = ''
                        self.nav_goal_dispatched = False
                        self.pending_goal_pose = None
                        self.result_nav_info = None
                        self.ensure_nav2 = False
                        self._moving_wait_logged = False
                        self._state = ILGP_State.WAIT_TRIGGER
                    elif self.latest_nav_result in ('canceled', 'failed: aborted by nav2'):
                        self.get_logger().warn(
                            f"Navigation ended with result '{self.latest_nav_result}', back to WAIT_TRIGGER"
                        )
                        self.latest_nav_result = ''
                        self.latest_nav_status = ''
                        self.latest_nav_feedback = ''
                        self.nav_goal_dispatched = False
                        self.pending_goal_pose = None
                        self.result_nav_info = None
                        self.ensure_nav2 = False
                        self._state = ILGP_State.WAIT_TRIGGER
                    elif self.latest_nav_result.startswith('failed:') or self.latest_nav_result.startswith('finished_with_status_'):
                        self.get_logger().warn(
                            f"Navigation failed with result '{self.latest_nav_result}', back to WAIT_TRIGGER"
                        )
                        self.latest_nav_result = ''
                        self.latest_nav_status = ''
                        self.latest_nav_feedback = ''
                        self.nav_goal_dispatched = False
                        self.pending_goal_pose = None
                        self.result_nav_info = None
                        self.ensure_nav2 = False
                        self._state = ILGP_State.WAIT_TRIGGER
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
