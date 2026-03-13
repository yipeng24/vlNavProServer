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
You are a waypoint decision module for instruction-guided visual navigation.

Instruction:
"{instruction}"

Input:
- 4 sequential egocentric RGB frames (Frame 1 oldest, Frame 4 latest)
- Frame size: {W} x {H}

Goal:
Predict the next local target pixel (u, v) in Frame 4, or determine that the instruction is already completed or currently impossible.

You must jointly use:
1. the instruction semantics,
2. the temporal change across all 4 frames,
3. the visible scene structure in Frame 4.

## Decision Process
First determine one of three states:
- "finish": the instructed subtask has already been visually completed,
- "move": the robot should continue moving,
- "noway": the instructed subtask is impossible from the current visual scene.

## Waypoint policy
If state is "move", the target pixel must satisfy all of the following:
- on visible walkable floor,
- locally reachable in the next step,
- advances the instruction,
- temporally consistent with the robot's recent motion,
- avoids obstacles, walls, furniture, door panels, and image borders,
- prefers safe, central, stable floor regions.

The target should be a LOCAL waypoint, not a final destination.

## Conservative finish rule
Only output "finish" if the instruction is truly completed in the latest visual state.
Seeing the goal is not enough.
Facing the goal is not enough.
Partially entering the goal region is not enough unless the action is clearly completed.

## noway rule
Only output "noway" if there is clear visual evidence that the instruction cannot be executed from here.

## Coordinate constraints
- u, v must be integers
- 0 <= u <= {W-1}
- 0 <= v <= {H-1}
- clip to valid range if necessary

## Output
Return exactly one JSON object and nothing else:

{{"sta":"finish","uv":"(-1,-1)","reason":"..."}}
or
{{"sta":"move","uv":"(u,v)","reason":"..."}}
or
{{"sta":"noway","uv":"(-1,-1)","reason":"..."}}

The reason must briefly mention:
- what visual evidence you used,
- why the state is correct,
- why the chosen point is the best next local floor target if moving.
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
            m2 = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", str(obj["uv"]))
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
            if uv is None and sta == "finish":
                uv = (-1, -1)
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
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/03.jpg",
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/04.jpg",
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/05.jpg",
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/10.jpg",
    ]

    img_paths = [os.path.expanduser(p) for p in img_paths]

    frames = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        frames.append(img)

    result = client.infer_vlm(
        "go out of the door and into the corridor",
        frames
    )
    print("Result:", result)
