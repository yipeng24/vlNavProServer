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
