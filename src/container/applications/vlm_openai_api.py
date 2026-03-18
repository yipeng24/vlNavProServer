import os
import time
import json
import re
import base64
import cv2

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def build_prompt(size_wh, used_frames, instruction: str) -> str:
    W, H = size_wh
    prompt = f"""
You are a waypoint decision module for instruction-guided navigation.

Instruction: "{instruction}"

Input:
- A chronological history of first-person RGB frames from the start of the current instruction until now.
- Total history length: 10 frames, but the actual number of frames provided may be fewer.
- A representative subset of {used_frames} frames is provided in order.
- The last frame is the current frame.
- Frame size: {W} x {H}

Task:
Decide whether the current instruction is already completed, still requires movement, or is currently impossible.
If movement is needed, predict one next local target pixel (u, v) in the current frame. In order to use as less steps as possible to complete the instruction, try to predict the waypoint that is closest to the final goal, rather than just the nearest one.
But notice that the predicted target must be on visible walkable floor, locally reachable, safe, and helpful for completing the current instruction. Do not predict targets that are on obstacles, far away, or not helpful for completing the instruction.

Rules:
1. First judge whether the CURRENT instruction itself is already satisfied using the full history and current frame.
2. Do not delay completion just because further movement is possible after the instruction is already satisfied.
3. If the current instruction is already satisfied, output "finish" instead of "move".
4. If output is "move", the target must be on visible walkable floor, locally reachable, safe, and helpful for completing the current instruction. Trying not to predict targets that are on obstacles.
5. Output "noway" only if the instruction cannot be executed from the current scene.

Point choosen rules:
1. The point must be on visible walkable floor, locally reachable, safe, and helpful.
2. u less than {W/2} means turning left, u greater than {W/2} means turning right. 
3. w higher measns closer, w lower means farther. But do not choose too far that is not reachable.
Try to predict points that can complete the instruction as soon as possible, rather than just the nearest one.

Output exactly one JSON object:
{{"sta":"finish","uv":"(-1,-1)","reason":"..."}}
or
{{"sta":"move","uv":"(u,v)","reason":"..."}}
or
{{"sta":"noway","uv":"(-1,-1)","reason":"..."}}
"""
    return prompt.strip()


class VLMClient:
    def __init__(self):
        self.client = None
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.isIDLE = True
        self.init_error = None

        if load_dotenv is not None:
            load_dotenv()

        if OpenAI is None:
            self.init_error = (
                "Python package 'openai' is not installed. "
                "Install it to enable VLM inference."
            )
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.init_error = "OPENAI_API_KEY not found in environment variables or .env"
            return

        self.client = OpenAI(api_key=api_key)

    def _img_to_data_url(self, img_bgr):
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            raise RuntimeError("cv2.imencode failed")

        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _build_input(self, instruction: str, rgb_frames_bgr: list):
        if not isinstance(rgb_frames_bgr, list) or len(rgb_frames_bgr) == 0:
            return []

        # 用最后一帧真实尺寸，别写死 640x480，不然容易抽风
        h, w = rgb_frames_bgr[-1].shape[:2]
        prompt = build_prompt(size_wh=(w, h), used_frames=len(rgb_frames_bgr), instruction=instruction)

        content = [
            {
                "type": "input_text",
                "text": prompt
            }
        ]

        for idx, img in enumerate(rgb_frames_bgr):
            data_url = self._img_to_data_url(img)
            content.append({
                "type": "input_text",
                "text": f"Frame {idx + 1}"
            })
            content.append({
                "type": "input_image",
                "image_url": data_url
            })

        return content

    def _parse_sta_uv(self, text: str):
        obj = None
        try:
            obj = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    obj = None

        if not isinstance(obj, dict):
            return None, None, None

        sta = obj.get("sta", None)
        reason = obj.get("reason", None)

        uv = None
        if "uv" in obj:
            m2 = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", str(obj["uv"]))
            if m2:
                uv = (int(m2.group(1)), int(m2.group(2)))
        elif "u" in obj and "v" in obj:
            try:
                uv = (int(obj["u"]), int(obj["v"]))
            except Exception:
                uv = None

        return sta, uv, reason

    def infer_vlm(self, instruction: str, rgb_frames_bgr: list):
        if self.client is None:
            return {
                "ok": False,
                "sta": "idle",
                "uv": None,
                "elapsed_s": 0.0,
                "raw_text": "",
                "error": self.init_error or "OpenAI client is unavailable",
            }

        if not isinstance(rgb_frames_bgr, list) or len(rgb_frames_bgr) == 0:
            return {
                "ok": False,
                "sta": "idle",
                "uv": None,
                "elapsed_s": 0.0,
                "raw_text": "",
                "error": "empty frames"
            }

        self.isIDLE = False
        t0 = time.perf_counter()

        try:
            input_content = self._build_input(instruction, rgb_frames_bgr)

            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": input_content
                    }
                ]
            )

            elapsed_s = time.perf_counter() - t0
            raw_text = getattr(response, "output_text", "") or ""

            sta, uv, reason = self._parse_sta_uv(raw_text)

            if uv is None and sta in ["finish", "noway"]:
                uv = (-1, -1)

            if uv is None:
                self.isIDLE = True
                return {
                    "ok": False,
                    "sta": sta,
                    "uv": None,
                    "reason": reason,
                    "elapsed_s": elapsed_s,
                    "raw_text": raw_text,
                    "error": "failed to parse uv",
                }

            self.isIDLE = True
            return {
                "ok": True,
                "sta": sta,
                "uv": uv,
                "reason": reason,
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
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/02.jpg",
        # "~/projects/vlNavProServer/imgs/go_out_of_the_door/03.jpg",
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/05.jpg",
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/07.jpg",
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/09.jpg",
        "~/projects/vlNavProServer/imgs/go_out_of_the_door/11.jpg",
        # "~/projects/vlNavProServer/imgs/go_out_of_the_door/12.jpg",
    ]
    img_paths = [os.path.expanduser(p) for p in img_paths]

    frames = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        frames.append(img)

    result = client.infer_vlm(
        "go out of the door and turn left till you are in the corridor",
        # "turn left into the corridor",
        frames
    )
    image_uv = result.get("uv", (-1, -1))
    print("Result:", result)
    vis_img = cv2.imread(img_paths[-1])
    if vis_img is None:
        raise FileNotFoundError(f"Failed to read image: {img_paths[-1]}")

    if isinstance(image_uv, (tuple, list)) and len(image_uv) == 2:
        u, v = int(image_uv[0]), int(image_uv[1])
        h, w = vis_img.shape[:2]
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(vis_img, (u, v), 10, (0, 255, 0), 2)

    output_path = os.path.expanduser("~/projects/vlNavProServer/tmp/vlm_openai_result.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    # cv2.imshow("VLM Result", vis_img)
    cv2.waitKey(0)
    print("Image UV:", image_uv)
    # print("Annotated image saved to:", output_path)
