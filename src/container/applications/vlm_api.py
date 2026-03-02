import os
from google import genai
from google.genai import types
import time
import json
import re
import cv2
from dotenv import load_dotenv
def build_prompt(size_wh, instruction: str) -> str:
    W, H = size_wh
    prompt = \
f"""
You are a path-planning navigation robot. 
Your current task is to execute the instruction: "{instruction}".

### Input Context:
- You are provided with 4 sequential image frames (Frame 1 is the oldest, Frame 4 is the latest).
- Image Dimensions: Width = {W}, Height = {H}.
- Your goal: Based on the temporal movement across these frames, select the next target pixel coordinate (u, v) in "Frame 4".

### Strict Constraints:
1. **Walkable Surface**: The point must be on the ground (navigable area). Do not place it on walls, ceilings, obstacles, or in the air.
2. **Coordinate Boundaries**: 
   - The u-coordinate must be an integer: 0 ≤ u ≤ {W-1}.
   - The v-coordinate must be an integer: 0 ≤ v ≤ {H-1}.
   - **CRITICAL**: If your calculated point is near the edge, you must clip it to ensure it does not exceed these maximum values (e.g., if W=640, u cannot be 640; it must be 639 or less).
3. **Status Logic (sta)**:
   - "finish": The instruction is fully completed.
   - "move": Further movement is required to reach the goal.
   - "noway": The instruction is impossible to fulfill from the current position.

### Output Format:
Output ONLY a valid JSON object. Do not include any conversational text or explanations.

{{"sta": "move", "uv": "(u,v)"}}
"""
    return prompt.strip()

class VLMClient:
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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
                model="gemini-2.5-flash",
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