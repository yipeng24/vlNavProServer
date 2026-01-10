# vlm_toolkit/inferencer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol
import numpy as np

@dataclass
class InferenceRequest:
    prompt: str
    rgb_frames_bgr: List[np.ndarray]          # history frames
    depth_frames: Optional[List[np.ndarray]]  # optional

class VLMBackend(Protocol):
    def infer(self, req: InferenceRequest) -> str:
        """Return raw model text."""
        ...

class VLMInferencer:
    def __init__(self, backend: VLMBackend):
        self.backend = backend

    def run(self, req: InferenceRequest) -> str:
        return self.backend.infer(req)

import json

class DummyBackend:
    """For pipeline testing (no real model)."""
    def infer(self, req: InferenceRequest) -> str:
        h, w = req.rgb_frames_bgr[-1].shape[:2]
        # return center-bottom-ish point
        out = {"u": w // 2, "v": int(h * 0.8), "update": True, "confidence": 0.1}
        return json.dumps(out)