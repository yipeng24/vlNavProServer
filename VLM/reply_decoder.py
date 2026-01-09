# vlm_toolkit/reply_decoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict
import json
import re

from .types import VLMOutput

@dataclass
class DecodeConfig:
    clamp_to_range: bool = True
    allow_fallback_regex: bool = True  # 如果不是纯json，用regex兜底
    default_update: bool = True

class ReplyDecoder:
    def __init__(self, width: int, height: int, cfg: Optional[DecodeConfig] = None):
        self.width = int(width)
        self.height = int(height)
        self.cfg = cfg or DecodeConfig()

    def decode(self, text: str) -> VLMOutput:
        raw = text or ""
        data = self._extract_json(raw)

        if data is None:
            raise ValueError(f"VLM reply decode failed. Raw text: {raw[:200]}")

        u = int(data.get("u"))
        v = int(data.get("v"))
        finish = bool(data.get("finish", self.cfg.default_update))

        if self.cfg.clamp_to_range:
            u = max(0, min(self.width - 1, u))
            v = max(0, min(self.height - 1, v))

        return VLMOutput(u=u, v=v, finish=finish, raw_text=raw, meta={"parsed": data})

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        # 1) 直接尝试整体json
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "u" in obj and "v" in obj:
                return obj
        except Exception:
            pass

        # 2) 从文本里找第一个 { ... } 片段
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "u" in obj and "v" in obj:
                return obj
        except Exception:
            return None
