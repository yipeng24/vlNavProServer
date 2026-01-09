# -*- coding: utf-8 -*-
"""
image_pooling.py

A thread-safe ring buffer for (RGB, Depth) frames from RealSense.
- Accepts compressed RGB and compressed depth (bytes).
- Keeps a deque as ring buffer.
- Provides:
    - get_latest_for_vlm(): latest 4 frames (paired)
    - get_latest_for_display(): latest 1 frame

Design notes:
- RGB and Depth may arrive asynchronously; we pair them by timestamp tolerance.
- For low FPS (1-2Hz), we keep a small time_tolerance and allow fallback pairing.
- The output frames include decoded numpy arrays (BGR for RGB, depth as uint16/float32 depending on decode).
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import cv2


@dataclass
class Frame:
    """A paired RGB+Depth frame."""
    stamp: float  # seconds (wall time or ROS time converted to float seconds)
    rgb_bgr: np.ndarray          # HxWx3 uint8 (BGR)
    depth: Optional[np.ndarray]  # HxW (uint16 or float32) or None
    rgb_stamp: Optional[float] = None
    depth_stamp: Optional[float] = None


class ImagePool:
    """
    Thread-safe ring buffer that pairs RGB and depth by timestamp.

    Typical usage (ROS2 callbacks):
        pool = ImagePool(vlm_window=4, maxlen=32)

        def rgb_cb(msg):
            pool.push_rgb_compressed(msg.data, stamp=msg.header.stamp.sec + ...)

        def depth_cb(msg):
            pool.push_depth_compressed(msg.data, stamp=...)

        latest4 = pool.get_latest_for_vlm()
        latest1 = pool.get_latest_for_display()
    """

    def __init__(
        self,
        vlm_window: int = 4,
        maxlen: int = 32,
        time_tolerance_s: float = 0.08,
        allow_unpaired_depth: bool = False,
        decode_rgb_to_bgr: bool = True,
    ):
        """
        Args:
            vlm_window: how many latest frames for VLM (default 4).
            maxlen: ring buffer capacity.
            time_tolerance_s: RGB/Depth pairing tolerance in seconds.
            allow_unpaired_depth: if True, can create frames with depth only (rarely needed).
            decode_rgb_to_bgr: if True, decode RGB compressed bytes to BGR np.ndarray via cv2.
        """
        self.vlm_window = int(vlm_window)
        self.buffer: Deque[Frame] = deque(maxlen=int(maxlen))

        self.time_tol = float(time_tolerance_s)
        self.allow_unpaired_depth = bool(allow_unpaired_depth)
        self.decode_rgb_to_bgr = bool(decode_rgb_to_bgr)

        self._lock = threading.Lock()

        # Hold "latest unpaired" messages until they can be paired.
        # For low fps, keep only a few.
        self._pending_rgb: Deque[Tuple[float, bytes]] = deque(maxlen=10)
        self._pending_depth: Deque[Tuple[float, bytes]] = deque(maxlen=10)

        # Stats / debug
        self._last_push_time = 0.0

    # -------------------------
    # Public push APIs
    # -------------------------
    def push_rgb_compressed(self, data: bytes, stamp: Optional[float] = None) -> None:
        """Push one compressed RGB image."""
        if stamp is None:
            stamp = time.time()
        with self._lock:
            self._pending_rgb.append((float(stamp), data))
            self._try_pair_locked()

    def push_depth_compressed(self, data: bytes, stamp: Optional[float] = None) -> None:
        """Push one compressed depth image (recommended: PNG-compressed 16UC1)."""
        if stamp is None:
            stamp = time.time()
        with self._lock:
            self._pending_depth.append((float(stamp), data))
            self._try_pair_locked()

    # -------------------------
    # Public read APIs
    # -------------------------
    def get_latest_for_display(self) -> Optional[Frame]:
        """Return the latest paired frame for real-time display."""
        with self._lock:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def get_latest_for_vlm(self) -> List[Frame]:
        """
        Return latest `vlm_window` frames.
        If fewer frames exist, return as many as available.
        Always returns in chronological order (old -> new).
        """
        with self._lock:
            if not self.buffer:
                return []
            k = min(self.vlm_window, len(self.buffer))
            frames = list(self.buffer)[-k:]
            return frames

    def size(self) -> int:
        with self._lock:
            return len(self.buffer)

    def clear(self) -> None:
        with self._lock:
            self.buffer.clear()
            self._pending_rgb.clear()
            self._pending_depth.clear()

    # -------------------------
    # Internal: pairing logic
    # -------------------------
    def _try_pair_locked(self) -> None:
        """
        Try to pair pending RGB and Depth by nearest timestamp within tolerance.
        Called under lock.
        """
        # Fast exit
        if not self._pending_rgb and not self._pending_depth:
            return

        # Greedy pairing:
        # - take the earliest RGB, find closest depth within tol
        # - or take earliest depth, find closest rgb within tol
        # whichever leads to a match.
        made_progress = True
        while made_progress:
            made_progress = False

            # Prefer RGB-driven pairing (typical: you want RGB as primary)
            if self._pending_rgb and self._pending_depth:
                rgb_stamp, rgb_bytes = self._pending_rgb[0]
                j = self._find_closest_index(self._pending_depth, rgb_stamp, self.time_tol)
                if j is not None:
                    depth_stamp, depth_bytes = self._pending_depth[j]
                    # pop rgb0 and depth[j]
                    self._pending_rgb.popleft()
                    self._pop_deque_index(self._pending_depth, j)
                    self._append_frame_locked(
                        rgb_stamp=rgb_stamp,
                        rgb_bytes=rgb_bytes,
                        depth_stamp=depth_stamp,
                        depth_bytes=depth_bytes,
                        pair_stamp=(rgb_stamp + depth_stamp) * 0.5,
                    )
                    made_progress = True
                    continue

            # If allow_unpaired_depth, we can also push depth-only frames
            if self.allow_unpaired_depth and self._pending_depth and not self._pending_rgb:
                depth_stamp, depth_bytes = self._pending_depth.popleft()
                # Depth-only frame: rgb=None not allowed by Frame, so skip or create dummy.
                # In practice, better to NOT enable allow_unpaired_depth.
                # We'll just keep pending until rgb arrives.
                pass

            # Cleanup too-old pending entries to avoid memory growth in weird sync situations
            self._cleanup_pending_locked()

    def _append_frame_locked(
        self,
        rgb_stamp: float,
        rgb_bytes: bytes,
        depth_stamp: float,
        depth_bytes: bytes,
        pair_stamp: float,
    ) -> None:
        rgb_bgr = self._decode_rgb(rgb_bytes)
        depth = self._decode_depth(depth_bytes)

        frame = Frame(
            stamp=float(pair_stamp),
            rgb_bgr=rgb_bgr,
            depth=depth,
            rgb_stamp=float(rgb_stamp),
            depth_stamp=float(depth_stamp),
        )
        self.buffer.append(frame)
        self._last_push_time = time.time()

    def _decode_rgb(self, rgb_bytes: bytes) -> np.ndarray:
        """
        Decode compressed RGB to BGR uint8.
        Works for JPEG/PNG compressed streams.
        """
        arr = np.frombuffer(rgb_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise ValueError("Failed to decode RGB compressed image (cv2.imdecode returned None).")
        return img

    def _decode_depth(self, depth_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode compressed depth. Recommended encoding: PNG of 16UC1 depth (millimeters).
        Using IMREAD_UNCHANGED to preserve uint16.
        """
        arr = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if depth is None:
            # depth optional (but for your project depth is needed for backprojection)
            raise ValueError("Failed to decode Depth compressed image (cv2.imdecode returned None).")
        # depth can be uint16 (good), or sometimes 8UC1 if badly encoded
        return depth

    @staticmethod
    def _find_closest_index(
        dq: Deque[Tuple[float, bytes]],
        target_t: float,
        tol: float
    ) -> Optional[int]:
        """
        Find index of element in dq whose timestamp is closest to target_t within tol.
        Return None if no element satisfies |t - target_t| <= tol.
        """
        best_i = None
        best_dt = None
        for i, (t, _) in enumerate(dq):
            dt = abs(t - target_t)
            if dt <= tol and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_i = i
        return best_i

    @staticmethod
    def _pop_deque_index(dq: Deque, idx: int) -> None:
        """Remove deque element by index (small dq only)."""
        if idx == 0:
            dq.popleft()
            return
        if idx == len(dq) - 1:
            dq.pop()
            return
        # general
        tmp = deque()
        for i in range(len(dq)):
            item = dq.popleft()
            if i != idx:
                tmp.append(item)
        dq.extend(tmp)

    def _cleanup_pending_locked(self) -> None:
        """
        Keep pending queues sane:
        - If oldest pending is too old compared with newest opposite stream, drop it.
        This avoids deadlocks when one stream stops.
        """
        # If one side is empty, nothing to compare.
        if not self._pending_rgb or not self._pending_depth:
            # Also cap by maxlen already, so nothing huge.
            return

        # If the oldest rgb is far earlier than newest depth, drop it (out of pairing window).
        newest_depth_t = self._pending_depth[-1][0]
        while self._pending_rgb and (newest_depth_t - self._pending_rgb[0][0]) > (5.0 * self.time_tol):
            self._pending_rgb.popleft()

        newest_rgb_t = self._pending_rgb[-1][0]
        while self._pending_depth and (newest_rgb_t - self._pending_depth[0][0]) > (5.0 * self.time_tol):
            self._pending_depth.popleft()
