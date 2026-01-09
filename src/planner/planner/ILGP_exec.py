#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ILGP_exec.py

Executor class (NOT a ROS node).
Runs a blocking user-input loop in its own thread:
- wait for instruction
- fetch latest frames
- VLM infer -> (u,v)
- backproject with depth -> camera 3D
- TF camera->base_footprint
- compute odom goal
- send NavigateToPose action
"""

import sys
import time
from typing import Optional, List

from vlm_toolkit.prompt_builder import PromptBuilder
from vlm_toolkit.reply_decoder import ReplyDecoder
from vlm_toolkit.inferencer import VLMInferencer, InferenceRequest, DummyBackend  # TODO replace backend


class ILGPExec:
    def __init__(self, node):
        """
        Args:
            node: ILGPNode instance (ROS2 node), providing:
                - get_latest_frames()
                - get_intrinsics()
                - pixel_to_3d_in_camera()
                - transform_point()
                - compute_goal_in_odom()
                - send_waypoint()
                - logger via node.get_logger()
        """
        self.node = node

        self.prompt_builder = PromptBuilder()
        self.backend = DummyBackend()  # TODO: replace with real VLM backend
        self.inferencer = VLMInferencer(backend=self.backend)

        self.decoder: Optional[ReplyDecoder] = None

    def run_loop(self):
        log = self.node.get_logger()
        log.info("ILGPExec thread started. Type instruction in terminal. (quit/exit to stop)")

        while self.node._alive and self.node.context.ok():
            try:
                sys.stdout.write("\nInstruction> ")
                sys.stdout.flush()
                instr = sys.stdin.readline()
                if not instr:
                    time.sleep(0.05)
                    continue
                instr = instr.strip()
                if instr == "":
                    continue
                if instr.lower() in ("quit", "exit", "q"):
                    log.info("User requested exit.")
                    self.node._alive = False
                    break

                self._one_step(instr)

            except Exception as e:
                log.error(f"ILGPExec loop error: {e}")
                time.sleep(0.2)

        log.info("ILGPExec thread stopped.")

    def _one_step(self, instruction: str):
        log = self.node.get_logger()

        frames = self.node.get_latest_frames()
        if len(frames) == 0:
            log.warn("No frames yet. Wait for camera topics...")
            return

        latest = frames[-1]
        if latest.depth is None:
            log.warn("Latest frame has no depth paired yet. Wait for depth topic...")
            return

        rgb_list = [f.rgb_bgr for f in frames]
        depth_latest = latest.depth
        h, w = rgb_list[-1].shape[:2]

        intr = self.node.get_intrinsics()
        if intr is None:
            log.warn("No CameraInfo yet. Wait for /camera_info...")
            return

        # Build prompt (images passed to backend via req)
        prompt = self.prompt_builder.build(instruction=instruction, width=w, height=h)
        raw = self.inferencer.run(InferenceRequest(prompt=prompt, rgb_frames_bgr=rgb_list, depth_frames=None))

        if self.decoder is None:
            self.decoder = ReplyDecoder(width=w, height=h)

        out = self.decoder.decode(raw)
        u, v = out.u, out.v
        self.node.set_last_uv(u, v)  # 让显示线程能画出这个点
        log.info(f"VLM output: u={u}, v={v}, update={out.update}, conf={out.confidence}")

        # Backproject pixel -> camera 3D
        x_cam, y_cam, z_cam = self.node.pixel_to_3d_in_camera(u, v, depth_latest, intr)
        log.info(f"Backproject: camera point = ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")

        # TF: camera -> base_footprint
        x_base, y_base, z_base = self.node.transform_point(
            x_cam, y_cam, z_cam,
            from_frame=self.node.camera_frame,
            to_frame=self.node.base_frame,
            stamp=None
        )
        log.info(f"TF to base: ({x_base:.3f}, {y_base:.3f}, {z_base:.3f})")

        # Compute odom goal from relative base displacement (dx,dy)
        goal_pose = self.node.compute_goal_in_odom(dx_base=float(x_base), dy_base=float(y_base))

        # Send Nav2 action
        _ = self.node.send_waypoint(goal_pose)
        log.info("Nav2 goal sent (async).")
