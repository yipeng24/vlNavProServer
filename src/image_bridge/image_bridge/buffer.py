#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FramePack:
    stamp_ns: int
    rgb_bgr: np.ndarray          # OpenCV BGR uint8
    depth: np.ndarray            # depth raw (often uint16/float32), keep as-is


class ImageRingBuffer:
    def __init__(self, maxlen: int = 30, sync_tolerance_ms: float = 80.0):
        self._buf: Deque[FramePack] = deque(maxlen=maxlen)

        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_rgb_ns: Optional[int] = None

        self._latest_depth: Optional[np.ndarray] = None
        self._latest_depth_ns: Optional[int] = None

        self._tol_ns: int = int(sync_tolerance_ms * 1e6)

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
        pack = FramePack(
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

    def get_latest(self, n: int = 1) -> List[FramePack]:
        if n <= 0:
            return []
        n = min(n, len(self._buf))
        return list(self._buf)[-n:]

