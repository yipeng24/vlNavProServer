#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import List, Optional

import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np

from image_pool.buffer import ImageRingBuffer, ImageFramePack
from rclpy.qos import qos_profile_sensor_data
from interfaces.srv import SaveLatest


def stamp_to_ns(msg: CompressedImage) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class ImageBridgeNode(Node):
    def __init__(self, ring: ImageRingBuffer = None):
        super().__init__('image_bridge_node')

        self.rgb_topic = '/net/color/image_rgb_compressed_2hz'
        self.depth_topic = '/net/depth/image_depth_compressed_2hz'
        self.buffer_maxlen = 30                 # ring size
        self.sync_tolerance_ms = 200          # rgb-depth 配对容忍
        self.save_dir = os.path.expanduser('~/image_bridge_saved')
        self.rgb_ext = "jpg"          # "jpg" / "png"
        self.rgb_jpg_quality = 90     # 1~100
        self.rgb_img=None
        # flip（如果相机倒装，按需打开）
        self.flip_rgb = True
        self.flip_depth = True
        self.flip_code = -1  # -1: both, 0: vertical, 1: horizontal

        self.enable_viewer = False
        self.viewer_hz = 10.0
        self.viewer_window = "image_bridge_latest_rgb"
        self.viewer_scale = 1.0  # >1 放大，<1 缩小

        # =========================
        self.bridge = CvBridge()
        self.ring = ImageRingBuffer(
            maxlen=self.buffer_maxlen,
            sync_tolerance_ms=self.sync_tolerance_ms
        )

        self.sub_rgb = self.create_subscription(
            CompressedImage, self.rgb_topic, self.on_rgb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            CompressedImage, self.depth_topic, self.on_depth, qos_profile_sensor_data
        )
        self.save_srv = self.create_service(SaveLatest, '/save_latest', self.on_save_latest)

        self.get_logger().info(f"Subscribed RGB : {self.rgb_topic}")
        self.get_logger().info(f"Subscribed Depth: {self.depth_topic}")
        self.get_logger().info(f"Buffer maxlen={self.buffer_maxlen}, tol={self.sync_tolerance_ms}ms")

        # 可选：定时打印状态
        # self.timer = self.create_timer(2.0, self.on_timer)

        # viewer timer
        self._viewer_timer: Optional[rclpy.timer.Timer] = None
        if self.enable_viewer:
            period = 1.0 / max(1e-6, float(self.viewer_hz))
            self._viewer_timer = self.create_timer(period, self._on_viewer)
            cv2.namedWindow(self.viewer_window, cv2.WINDOW_NORMAL)


    def on_save_latest(self, req, res):
        # self.get_logger().info(f"Service /save_latest called: n={req.n}, out_dir={req.out_dir}")
        try:
            prefixes = self.save_latest_to_disk(
                n=req.n,
                out_dir=req.out_dir,
                save_depth_png=req.save_depth_png,
                save_depth_npy=req.save_depth_npy,
                
            )
            res.ok = True
            res.prefixes = prefixes
            res.message = "ok"
            self.get_logger().info(f"Saved {len(prefixes)} packs to {req.out_dir}")
        except Exception as e:
            res.ok = False
            res.prefixes = []
            res.message = str(e)
            self.get_logger().error(f"Save latest failed: {e}")
        return res
    

    def on_timer(self):
        self.get_logger().info(f"ring size = {self.ring.size()}")

    def on_rgb(self, msg: CompressedImage):
        # self.get_logger().info(f"RGB msg stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"RGB cv_bridge failed: {e}")
            return

        if self.flip_rgb:
            import cv2
            bgr = cv2.flip(bgr, self.flip_code)

        self.ring.update_rgb(stamp_to_ns(msg), bgr)


    def on_depth(self, msg: CompressedImage):
        stamp_ns = stamp_to_ns(msg)
        # self.get_logger().info(
        #     f"DEPTH msg stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} format={msg.format}"
        # )
        try:
            if "compressedDepth" in msg.format:
                raw = bytes(msg.data)
                png_sig = b"\x89PNG\r\n\x1a\n"

                idx = raw.find(png_sig)
                if idx < 0:
                    # 给你更多信息方便继续定位
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

        self.ring.update_depth(stamp_ns, depth)
    # ========= viewer =========
    def _on_viewer(self):
        packs = self.ring.get_latest(1)
        if not packs:
            # still need waitKey to keep window responsive
            cv2.waitKey(1)
            return

        p = packs[-1]
        img = p.rgb_bgr

        # overlay
        text1 = f"stamp(ns): {p.stamp_ns}"
        text2 = f"ring: {self.ring.size()}/{self.buffer_maxlen}"
        vis = img.copy()
        cv2.putText(vis, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(vis, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if abs(self.viewer_scale - 1.0) > 1e-3:
            vis = cv2.resize(vis, None, fx=self.viewer_scale, fy=self.viewer_scale, interpolation=cv2.INTER_NEAREST)

        cv2.imshow(self.viewer_window, vis)
        cv2.waitKey(1)

    def destroy_node(self):
        try:
            if self.enable_viewer:
                cv2.destroyWindow(self.viewer_window)
        except Exception:
            pass
        return super().destroy_node()
    # ========== 对外接口（你后续主线程 import 这个 node 或 ring 来用）==========

    def save_latest(self, n: int = 1):
        saved = self.ring.save_latest(self.save_dir, n=n)
        self.get_logger().info(f"saved {len(saved)} packs into {self.save_dir}")
        return saved

    def get_latest(self, n: int = 1):
        return self.ring.get_latest(n)
    
    # ========= node 层写盘接口（buffer 不涉及）=========
    def save_latest_to_disk(
        self,
        n: int = 1,
        out_dir: str | None = None,
        save_depth_png: bool = True,
        save_depth_npy: bool = True,
    ) -> List[str]:
        """
        Save latest n packs to disk (called manually).
        Returns list of file prefixes.

        RGB: jpg/png
        Depth:
          - npy: always lossless
          - png: if uint16 -> lossless 16-bit; else -> for preview only (normalized to 8-bit)
        """
        out_dir = out_dir or self.save_dir
        os.makedirs(out_dir, exist_ok=True)

        packs = self.ring.get_latest(n)
        saved_prefixes: List[str] = []

        # jpg settings
        jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.rgb_jpg_quality)]

        for p in packs:
            prefix = os.path.join(out_dir, f"{p.stamp_ns}")
            rgb_path = f"{prefix}_rgb.{self.rgb_ext}"
            depth_png_path = f"{prefix}_depth.png"
            depth_npy_path = f"{prefix}_depth.npy"

            # save rgb
            if self.rgb_ext.lower() in ["jpg", "jpeg"]:
                cv2.imwrite(rgb_path, p.rgb_bgr, jpg_params)
            else:
                cv2.imwrite(rgb_path, p.rgb_bgr)

            # save depth
            if save_depth_npy:
                np.save(depth_npy_path, p.depth)

            if save_depth_png:
                d = p.depth
                if d.dtype == np.uint16:
                    cv2.imwrite(depth_png_path, d)
                else:
                    # preview-only normalization
                    d_vis = d.astype(np.float32)
                    mn, mx = float(np.nanmin(d_vis)), float(np.nanmax(d_vis))
                    if mx - mn < 1e-6:
                        d8 = np.zeros_like(d_vis, dtype=np.uint8)
                    else:
                        d8 = ((d_vis - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(depth_png_path, d8)

            saved_prefixes.append(prefix)

        self.get_logger().info(f"Saved {len(saved_prefixes)} packs to {out_dir}")
        return saved_prefixes

def main():
    rclpy.init()
    node = ImageBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()