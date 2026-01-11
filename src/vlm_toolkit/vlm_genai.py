import os
from google import genai
from google.genai import types
import time
import json
import re
import cv2
import sys

def build_prompt(size_wh, instruction: str) -> str:
    W, H = size_wh
    prompt = f"""
你现在是一个规划路径点的导航机器人。
现在你要执行的指令是“{instruction}”。

我给你参考的是现在四帧(旧的排在前, 第四张是最新)的图像，图像尺寸是 {W}x{H}。
请你根据四帧的变化，选择“在第四张图像里”下一步应该到达的像素坐标(u,v)。

要求：
1) 输出的点必须在地面上（可行走区域），不能在墙上、门上、桌子等物体上，也不能在空中。
2) u 范围 0-{W-1}，v 范围 0-{H-1}，必须是整数。
3) 如果你认为这个指令已经完成：sta="finish"
   如果你认为还需要继续走：sta="move"
   如果你认为无法完成指令：sta="noway"

输出格式（只允许输出 JSON，不要输出任何解释文字）：
{{"sta":"move","uv":"(u,v)"}}

示例：
{{"sta":"move","uv":"(123,123)"}}
"""
    return prompt.strip()

class VLMClient:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GENAI_API_KEY", ""))

        # 你自己的 prompt 模板：要求输出 JSON，强烈建议固定格式，方便解析
        # 例如：{"uv":"(u,v)","reason":"..."} 或 {"u":123,"v":456}
        self.prompt_template = None
        self.isIDLE = True

    def part_from_bgr(self, bgr_img):
        """
        Convert BGR np.ndarray -> image bytes part for Gemini.
        """
        if bgr_img is None:
            return None

        # JPEG 更小更快；PNG更保真但大
        ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            raise RuntimeError("cv2.imencode failed")

        return types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")

    def _build_contents(self, instruction: str, rgb_frames_bgr: list):
        prompt = build_prompt(size_wh=(640, 480), instruction=instruction)

        img_parts = []
        for img in rgb_frames_bgr:
            img_parts.append(self.part_from_bgr(img))

        # 有些 SDK 允许 contents = [*img_parts, prompt]
        # 有些需要 contents = [{"role":"user","parts":[...]}]
        # 你当前代码风格：contents = [parts...] + [prompt]
        return img_parts + [prompt]

    def _parse_sta_uv(self, text: str):
        # 复用你之前的 JSON 提取逻辑
        obj = None
        try:
            obj = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                obj = json.loads(m.group(0))

        if not isinstance(obj, dict):
            return None, None

        sta = obj.get("sta", None)
        uv = None
        if "uv" in obj:
            m2 = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", str(obj["uv"]))
            if m2:
                uv = (int(m2.group(1)), int(m2.group(2)))
        elif "u" in obj and "v" in obj:
            uv = (int(obj["u"]), int(obj["v"]))

        return sta, uv

    def _uv_from_obj(self, obj):
        if obj is None:
            return None
        # {"uv":"(u,v)"}
        if isinstance(obj, dict) and "uv" in obj:
            uv = obj["uv"]
            if isinstance(uv, str):
                m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", uv)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
        return None

    def infer_vlm(self, instruction: str, rgb_frames_bgr: list):
        """
        输入：instruction + 4帧 BGR ndarray 列表（历史->最新）
        输出：dict:
          {
            "ok": True/False,
            "uv": (u,v) or None,
            "elapsed_s": float,
            "raw_text": str,
            "error": str (if any)
          }
        """
        if not isinstance(rgb_frames_bgr, list) or len(rgb_frames_bgr) == 0:
            return {"ok": False, "sta": "ldle", "uv": None, "elapsed_s": 0.0, "raw_text": "", "error": "empty frames"}

        contents = self._build_contents(instruction, rgb_frames_bgr)
        self.isIDLE = False
        t0 = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            elapsed_s = time.perf_counter() - t0

            raw_text = getattr(response, "text", "") or ""
            sta, uv = self._parse_sta_uv(raw_text)

            if uv is None:
                return {
                    "ok": False,
                    "sta": sta,
                    "uv": None,
                    "elapsed_s": elapsed_s,
                    "raw_text": raw_text,
                    "error": "failed to parse uv",
                }
            self.isIDLE = True
            return {
                "ok": True,
                "sta": sta,
                "uv": uv,
                "elapsed_s": elapsed_s,
                "raw_text": raw_text,
            }

        except Exception as e:
            elapsed_s = time.perf_counter() - t0
            self.isIDLE = True
            return {
                "ok": False,
                "uv": None,
                "elapsed_s": elapsed_s,
                "raw_text": "",
                "error": f"{type(e).__name__}: {e}",
            }









