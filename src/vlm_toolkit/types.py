# vlm_toolkit/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Any, Dict
import numpy as np

@dataclass
class VLMInput:
    instruction: str
    rgb_frames_bgr: List[np.ndarray]          # len=4 (or <=4)
    depth_frames: Optional[List[np.ndarray]]  # optional, same length
    image_size: tuple[int, int]               # (W, H)

@dataclass
class VLMOutput:
    u: int
    v: int
    finish: bool = True                # 执行完成则不更新: False
    raw_text: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
