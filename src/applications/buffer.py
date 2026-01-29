#!/usr/bin/env python3
from __future__ import annotations

import threading
import os
import time

from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ImageFramePack:
    stamp_ns: int
    rgb_bgr: np.ndarray          # OpenCV BGR uint8
    depth: np.ndarray            # depth raw (often uint16/float32), keep as-is


class ImageRingBuffer:
    def __init__(self, maxlen: int = 30, sync_tolerance_ms: float = 100.0):
        self.instance = self
        self.maxlen = maxlen
        self._buf: Deque[ImageFramePack] = deque(maxlen=maxlen)

        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_rgb_ns: Optional[int] = None

        self._latest_depth: Optional[np.ndarray] = None
        self._latest_depth_ns: Optional[int] = None

        self._tol_ns: int = int(sync_tolerance_ms * 1e6)

        self._lock = threading.Lock()

    def update_rgb(self, stamp_ns: int, rgb_bgr: np.ndarray) -> None:
        self._latest_rgb = rgb_bgr
        self._latest_rgb_ns = stamp_ns
        self._try_pair()

    def update_depth(self, stamp_ns: int, depth: np.ndarray) -> None:
        self._latest_depth = depth
        self._latest_depth_ns = stamp_ns
        self._try_pair()

    def _try_pair(self) -> None:
        if self._latest_rgb is None or self._latest_depth is None:
            return
        if self._latest_rgb_ns is None or self._latest_depth_ns is None:
            return

        dt = abs(self._latest_rgb_ns - self._latest_depth_ns)
        if dt > self._tol_ns:
            return

        stamp_ns = min(self._latest_rgb_ns, self._latest_depth_ns)
        pack = ImageFramePack(
            stamp_ns=stamp_ns,
            rgb_bgr=self._latest_rgb.copy(),
            depth=self._latest_depth.copy(),
        )
        self._buf.append(pack)

        # avoid generating duplicated packs when one topic bursts
        self._latest_rgb_ns = None
        self._latest_depth_ns = None


    def size(self) -> int:
        return len(self._buf)


    def get_latest(self, n: int = 1) -> List[ImageFramePack]:
        if n <= 0:
            return []
        n = min(n, len(self._buf))
        return list(self._buf)[-n:]


    def save_latest(self, out_dir: Optional[str]) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)

        with self._lock:
            if len(self._buf) == 0:
                return {"ok": False, "msg": "ring empty"}
            p = self._buf[-1]
            stamp_ns = getattr(p, "stamp_ns", int(time.time() * 1e9))
            rgb = getattr(p, "rgb_bgr", None)
            if rgb is None:
                return {"ok": False, "msg": "latest pack has no rgb_bgr"}
            rgb_copy = rgb.copy()

            prefix = os.path.join(out_dir, str(stamp_ns))
            rgb_path = f"{prefix}_rgb.jpg"

            ok = cv2.imwrite(rgb_path, rgb_copy)
            if not ok:
                return {"ok": False, "msg": f"cv2.imwrite failed: {rgb_path}", "stamp_ns": stamp_ns}
        return {
            "ok": True,
            "rgb_path": rgb_path,
            "stamp_ns": stamp_ns,
            "msg": "saved"
        }