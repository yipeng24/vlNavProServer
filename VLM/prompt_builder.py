# vlm_toolkit/prompt_builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class PromptConfig:
    # 强约束：只输出 JSON，便于解码
    force_json: bool = True
    # 输出约束：可达点在地面、像素范围等
    constrain_reachable_ground: bool = True
    # 历史帧数量提示（不一定要写死）
    history_n: int = 4

class PromptBuilder:
    def __init__(self, cfg: Optional[PromptConfig] = None):
        self.cfg = cfg or PromptConfig()

    def build(self, instruction: str, width: int, height: int) -> str:
        """
        这里不直接塞图片（图片由 inferencer 负责传给 VLM），这里只构造文字prompt。
        """
        rules = [
            f"Image size: width={width}, height={height}.",
            "You are a navigation robot. Choose the next waypoint pixel (u,v) on the ground.",
            "u is horizontal pixel coordinate in [0, width-1], v is vertical pixel coordinate in [0, height-1].",
        ]
        if self.cfg.constrain_reachable_ground:
            rules.append("The waypoint MUST be reachable and on the floor region (not on walls/doors).")

        output_spec = """
Return ONLY a JSON object in the following schema:
{
  "u": <int>,
  "v": <int>,
  "update": <bool>,
}
- "update" = false if the previous waypoint is not finished yet and you decide not to update.
No extra text, no markdown.
""".strip()

        prompt = "\n".join([
            "TASK:",
            instruction.strip(),
            "",
            "RULES:",
            "\n".join(f"- {r}" for r in rules),
            "",
            output_spec
        ])
        return prompt
