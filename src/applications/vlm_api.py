import os
from google import genai
from google.genai import types
import time
import json
import re
import cv2

def build_prompt(size_wh, instruction: str) -> str:
    W, H = size_wh
    prompt = \
f"""
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
        self.client = genai.Client(api_key="AIzaSyC_jK6Y2DNmiWZj1H-YEJeZeN1OnPnVQZc")
        self.isIDLE = True

    def _build_contents(self, instruction: str, rgb_frames_bgr: list):
        if not isinstance(rgb_frames_bgr, list) or len(rgb_frames_bgr) == 0:
            return []
        prompt = build_prompt(size_wh=(640, 480), instruction=instruction)

        img_parts = []
        for img in rgb_frames_bgr:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ok:
                raise RuntimeError("cv2.imencode failed")

            encodeImg=types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
            img_parts.append(encodeImg)

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
    

    def infer_vlm(self, instruction: str, rgb_frames_bgr: list):
        if not isinstance(rgb_frames_bgr, list) or len(rgb_frames_bgr) == 0:
            return {"ok": False, "sta": "ldle", "uv": None, "elapsed_s": 0.0, "raw_text": "", "error": "empty frames"}
        
        contents = self._build_contents(instruction, rgb_frames_bgr)
        self.isIDLE = False
        t0 = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                # model="gemini-robotics-er-1.5-preview",
                model="gemini-2.5-flash-lite",
                contents=contents,
                # config=types.GenerateContentConfig(
                #     temperature=0.2,
                #     thinking_config=types.ThinkingConfig(thinking_budget=0)
                # )
            )
            elapsed_s = time.perf_counter() - t0

            raw_text = getattr(response, "text", "") or ""
            sta, uv = self._parse_sta_uv(raw_text)
            if uv is None:
                return {
                    "ok": False, "sta": sta, "uv": None, "elapsed_s": elapsed_s, "raw_text": raw_text, "error": "failed to parse uv",
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

if __name__ == "__main__":
    client = VLMClient()
    img_paths = [
        "vlNavProServer\\imgs\\go_out_of_the_door\\12.jpg",

    ]
    result = client.infer_vlm("go out of the door and into the corridor", [cv2.imread(p) for p in img_paths])
    print("Result:", result)