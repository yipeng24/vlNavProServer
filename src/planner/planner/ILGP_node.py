#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from nav2_msgs.action import NavigateToPose

import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs  # noqa: F401

from image_pooling import ImagePool, Frame
from ILGP_exec import ILGPExec
import cv2
import numpy as np
import time

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class ILGPNode(Node):
    def __init__(self):
        super().__init__("ilgp_node")

        # ---------- config ----------
        self.rgb_topic = "/camera/color/image_raw/compressed"
        self.depth_topic = "/camera/aligned_depth_to_color/image_raw/compressed"
        self.camerainfo_topic = "/camera/color/camera_info"

        self.odom_frame = "odom"
        self.base_frame = "base_footprint"
        self.camera_frame = "camera_color_optical_frame"  # TODO: set to your actual frame

        self.depth_unit_scale = 0.001
        self.min_depth_m = 0.2
        self.max_depth_m = 6.0

        self.pool = ImagePool(vlm_window=4, maxlen=32, time_tolerance_s=0.2)

        self._intrinsics_lock = threading.Lock()
        self._intrinsics: Optional[CameraIntrinsics] = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # ---------- subs ----------
        self.create_subscription(CompressedImage, self.rgb_topic, self._rgb_cb, 10)
        self.create_subscription(CompressedImage, self.depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, self.camerainfo_topic, self._camerainfo_cb, 10)

        # ---------- exec thread ----------
        self._alive = True
        self._exec = ILGPExec(self)
        self._exec_thread = threading.Thread(target=self._exec.run_loop, daemon=True)
        self._exec_thread.start()
        # ---------- display thread ----------
        self.enable_display = True
        self.display_fps = 10  # 显示刷新频率，不影响相机1-2Hz
        self._uv_lock = threading.Lock()
        self._last_uv = None  # (u, v) from last inference

        self._disp_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._disp_thread.start()

        self.get_logger().info("ILGPNode started. Exec thread launched. Waiting for images/TF/Nav2...")

    # -------- callbacks --------
    def _rgb_cb(self, msg: CompressedImage):
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.pool.push_rgb_compressed(msg.data, stamp=stamp)

    def _depth_cb(self, msg: CompressedImage):
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.pool.push_depth_compressed(msg.data, stamp=stamp)

    def _camerainfo_cb(self, msg: CameraInfo):
        fx = float(msg.k[0]); fy = float(msg.k[4])
        cx = float(msg.k[2]); cy = float(msg.k[5])
        intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=int(msg.width), height=int(msg.height))
        with self._intrinsics_lock:
            self._intrinsics = intr

    # -------- public helpers --------
    def get_intrinsics(self) -> Optional[CameraIntrinsics]:
        with self._intrinsics_lock:
            return self._intrinsics

    def get_latest_frames(self) -> List[Frame]:
        return self.pool.get_latest_for_vlm()

    # -------- geometry pipeline --------
    def pixel_to_3d_in_camera(self, u: int, v: int, depth_img: np.ndarray, intr: CameraIntrinsics):
        h, w = depth_img.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            raise ValueError(f"Pixel out of bounds: u={u}, v={v}, depth_wh=({w},{h})")

        d = depth_img[v, u]
        if depth_img.dtype == np.uint16:
            z = float(d) * self.depth_unit_scale
        elif depth_img.dtype in (np.float32, np.float64):
            z = float(d)
        else:
            raise ValueError(f"Unsupported depth dtype: {depth_img.dtype} (prefer PNG 16UC1).")

        if not (self.min_depth_m <= z <= self.max_depth_m):
            raise ValueError(f"Depth out of range: z={z:.3f} m at (u,v)=({u},{v})")

        x = (float(u) - intr.cx) / intr.fx * z
        y = (float(v) - intr.cy) / intr.fy * z
        return (x, y, z)

    def transform_point(self, x: float, y: float, z: float, from_frame: str, to_frame: str, stamp=None):
        ps = PointStamped()
        ps.header.frame_id = from_frame
        ps.point.x = float(x); ps.point.y = float(y); ps.point.z = float(z)

        try:
            out = self.tf_buffer.transform(ps, to_frame, timeout=rclpy.duration.Duration(seconds=0.5))
            return (out.point.x, out.point.y, out.point.z)
        except TransformException as e:
            raise RuntimeError(f"TF transform failed {from_frame}->{to_frame}: {e}")

    def lookup_base_in_odom(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except TransformException as e:
            raise RuntimeError(f"TF lookup failed {self.odom_frame}<-{self.base_frame}: {e}")

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        return (tx, ty, yaw)

    def compute_goal_in_odom(self, dx_base: float, dy_base: float) -> PoseStamped:
        bx, by, byaw = self.lookup_base_in_odom()

        gx = bx + math.cos(byaw) * dx_base - math.sin(byaw) * dy_base
        gy = by + math.sin(byaw) * dx_base + math.cos(byaw) * dy_base

        qx, qy, qz, qw = yaw_to_quat(byaw)

        goal = PoseStamped()
        goal.header.frame_id = self.odom_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal
    # --show img---------
    def set_last_uv(self, u: int, v: int):
        with self._uv_lock:
            self._last_uv = (int(u), int(v))

    def get_last_uv(self):
        with self._uv_lock:
            return self._last_uv
    def _display_loop(self):
        """
        Display the latest RGB frame in a separate thread.
        Draw last inferred (u,v) if available.
        Press 'q' in the window to stop display (node keeps running).
        """
        if not self.enable_display:
            return

        win = "ILGP Live"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        period = 1.0 / max(1, int(self.display_fps))

        while self._alive and self.context.ok():
            frame = self.pool.get_latest_for_display()
            if frame is None:
                # still no image
                cv2.waitKey(1)
                time.sleep(period)
                continue

            img = frame.rgb_bgr
            if img is None:
                cv2.waitKey(1)
                time.sleep(period)
                continue

            vis = img.copy()

            # draw last uv if available
            uv = self.get_last_uv()
            if uv is not None:
                u, v = uv
                h, w = vis.shape[:2]
                u = max(0, min(w - 1, u))
                v = max(0, min(h - 1, v))
                cv2.circle(vis, (u, v), 6, (0, 255, 0), 2)
                cv2.putText(vis, f"uv=({u},{v})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # stop only display loop
                self.get_logger().info("Display stopped by user (pressed 'q').")
                break

            time.sleep(period)

        try:
            cv2.destroyWindow(win)
        except Exception:
            pass

        if frame.depth is not None:
            d = frame.depth
            if d.dtype == np.uint16:
                dm = d.astype(np.float32) * self.depth_unit_scale
            else:
                dm = d.astype(np.float32)
            dm = np.clip(dm, 0.0, 5.0)
            dm8 = (dm / 5.0 * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(dm8, cv2.COLORMAP_JET)
            cv2.imshow("ILGP Depth", depth_vis)

    # -------- nav2 action --------
    def send_waypoint(self, goal_pose: PoseStamped):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("Nav2 action server 'navigate_to_pose' not available.")
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        self.get_logger().info(
            f"Sending goal: odom x={goal_pose.pose.position.x:.3f}, y={goal_pose.pose.position.y:.3f}"
        )
        return self.nav_client.send_goal_async(goal_msg)

    def destroy_node(self):
        # stop exec thread
        self._alive = False
        super().destroy_node()


def main():
    rclpy.init()
    node = ILGPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._alive = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
