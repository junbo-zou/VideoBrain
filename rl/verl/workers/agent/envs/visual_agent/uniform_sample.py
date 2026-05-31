import re
import random
import time
import json
import decord
import numpy as np
from typing import List, Dict, Tuple, Any
from PIL import Image
from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents


class UniformSample(ToolBase):
    name = "uniform_sample"
    action_start = '<tool_call>'
    action_end = '</tool_call>'
    chat_template = """<|im_end|>
        <|im_start|>user
        {}<|im_end|>
        <|im_start|>assistant
        """

    def __init__(self, _name, _desc, _params, **kwargs):
        self.chatml_history = []
        self.multi_modal_data = None
        self.video_path = None
        self.fps = None
        self.total_frames = None
        self.vr = None

        self.height = 0
        self.width = 0
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        print(f"[DEBUG execute] Called with video_path={self.video_path}, total_frames={self.total_frames}")

        if not self.video_path:
            print("[DEBUG execute] Terminating: no video_path")
            return '', 0.0, True, {}

        action_block = extract_tool_call_contents(self.action_start, self.action_end, action_string)
        if not action_block:
            print(f"[DEBUG execute] Terminating: no action_block found in action_string='{action_string}'")
            return '', 0.0, True, {}
        action_block = action_block[-1]
        print(f"[DEBUG execute] Extracted action_block: '{action_block}'")

        # Parse JSON format: {"name": "uniform_sample", "arguments": {"start_frame": X, "end_frame": Y, "num_frame": Z}}
        try:
            action_data = json.loads(action_block.strip())
            print(f"[DEBUG execute] Parsed JSON action_data: {action_data}")
        except json.JSONDecodeError as e:
            print(f"[DEBUG execute] Terminating: JSON parsing failed with error: {e}")
            print(f"[DEBUG execute] Action block content: '{action_block}'")
            return '', 0.0, True, {}

        # Check if name is "uniform_sample"
        if action_data.get("name") != "uniform_sample":
            print(f"[DEBUG execute] Terminating: unknown tool name '{action_data.get('name')}'")
            return '', 0.0, True, {}

        # Extract arguments
        arguments = action_data.get("arguments")
        if not arguments or not isinstance(arguments, dict):
            print(f"[DEBUG execute] Terminating: invalid arguments format: {arguments}")
            return '', 0.0, True, {}

        start_frame = arguments.get("start_frame")
        end_frame = arguments.get("end_frame")
        num_frame = arguments.get("num_frame")

        # Validate parameters
        if start_frame is None or end_frame is None or num_frame is None:
            print(f"[DEBUG execute] Terminating: missing required parameters")
            return '', 0.0, True, {}

        try:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            num_frame = int(num_frame)
        except (ValueError, TypeError) as e:
            print(f"[DEBUG execute] Terminating: parameter type conversion failed: {e}")
            return '', 0.0, True, {}

        print(f"[DEBUG execute] Parsed parameters: start_frame={start_frame}, end_frame={end_frame}, num_frame={num_frame}")

        # Check num_frame constraint (>20 is format error) - following think_with_video error handling
        if num_frame > 20:
            print(f"[DEBUG execute] Terminating: num_frame({num_frame}) > 20 (format error)")
            return '', 0.0, True, {}

        # Validate frame range
        if start_frame < 0 or start_frame >= self.total_frames:
            print(f"[DEBUG execute] Terminating: invalid start_frame({start_frame}), total_frames={self.total_frames}")
            return '', 0.0, True, {}

        if end_frame > self.total_frames:
            print(f"[DEBUG execute] Adjusting end_frame from {end_frame} to {self.total_frames}")
            end_frame = self.total_frames

        if start_frame >= end_frame:
            print(f"[DEBUG execute] Terminating: start_frame({start_frame}) >= end_frame({end_frame})")
            return '', 0.0, True, {}

        if end_frame - start_frame < num_frame:
            print(f"[DEBUG execute] Terminating: frame range({end_frame - start_frame}) < num_frame({num_frame})")
            return '', 0.0, True, {}

        print(f"[DEBUG execute] Calling _uniform_sample_frames with start={start_frame}, end={end_frame}, num={num_frame}")
        user_msg, sampled_images = self._uniform_sample_frames(
            start_frame=start_frame,
            end_frame=end_frame,
            num_frame=num_frame
        )
        print(f"[DEBUG execute] _uniform_sample_frames returned: msg_len={len(user_msg)}, num_images={len(sampled_images)}")

        all_user_msg = self.chat_template.format(user_msg)
        if len(sampled_images) == 0:
            print("[DEBUG execute] Terminating: no images extracted (empty sampled_images)")
            return '', 0.0, True, {}

        print(f"[DEBUG execute] Success! Returning {len(sampled_images)} images with done=False")
        obs_dict = {
            "prompt": all_user_msg,
            "multi_modal_data": {
                "image": sampled_images
            }
        }
        return obs_dict, 0.0, False, {}

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.video_path = kwargs.get('video_path')
        self.fps = kwargs.get('fps')
        self.total_frames = kwargs.get('total_frames')
        self.height = kwargs.get('height')
        self.width = kwargs.get('width')
        self.vr = None
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data

        print(f"[DEBUG reset] Initializing UniformSample:")
        print(f"  video_path: {self.video_path}")
        print(f"  fps: {self.fps}")
        print(f"  total_frames: {self.total_frames}")
        print(f"  height: {self.height}, width: {self.width}")

        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width
        print(f"  Initial image dimensions from multi_modal_data: {self.width}x{self.height}")
        print(f"[DEBUG reset] Reset complete!")

    def _uniform_sample_frames(
            self,
            start_frame: int,
            end_frame: int,
            num_frame: int
    ) -> Tuple[str, List[Image.Image]]:
        """
        Uniformly sample num_frame frames from [start_frame, end_frame)

        Scaling rules:
        - num_frame <= 12: 640x360
        - 12 < num_frame <= 20: 448x252
        - num_frame > 20: should be rejected in execute()
        """
        # Determine target resolution based on num_frame
        if num_frame > 12:
            target_w, target_h = 448, 252
        else:
            target_w, target_h = 640, 360

        print(f"[DEBUG sample] Processing frames: start={start_frame}, end={end_frame}, num={num_frame}")
        print(f"[DEBUG sample] Target dimensions: {target_w}x{target_h}")

        try:
            if self.vr is None:
                print(f"[DEBUG sample] Initializing VideoReader for path: {self.video_path}")
                max_retries = 3
                base_delay = 2
                for attempt in range(max_retries):
                    try:
                        self.vr = decord.VideoReader(
                            self.video_path,
                            ctx=decord.cpu(0),
                            width=target_w,
                            height=target_h
                        )
                        print(f"[DEBUG sample] VideoReader initialized successfully, total frames in video: {len(self.vr)}")
                        break
                    except Exception as e:
                        if "Resource temporarily unavailable" in str(e) and attempt < max_retries - 1:
                            wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                            print(
                                f"WARN: [Attempt {attempt + 1}/{max_retries}] Failed to open video due to resource issue. "
                                f"Retrying in {wait_time:.2f} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(
                                f"ERROR: [Attempt {attempt + 1}/{max_retries}] Unrecoverable error or max retries reached. Raising exception.")
                            raise e
            else:
                # If VideoReader already exists but resolution might be different, reinitialize
                # This handles the case where previous call used different num_frame threshold
                current_width = self.vr[0].shape[1]
                current_height = self.vr[0].shape[0]
                if current_width != target_w or current_height != target_h:
                    print(f"[DEBUG sample] Resolution mismatch ({current_width}x{current_height} vs {target_w}x{target_h}), reinitializing VideoReader")
                    self.vr = decord.VideoReader(
                        self.video_path,
                        ctx=decord.cpu(0),
                        width=target_w,
                        height=target_h
                    )

            # Generate uniformly spaced frame indices
            # Use linspace to get evenly distributed indices between start_frame and end_frame
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frame, dtype=int).tolist()
            # Remove duplicates and sort
            frame_indices = sorted(list(set(frame_indices)))
            print(f"[DEBUG sample] Generated frame_indices: {frame_indices}")

            sampled_frames_array = self.vr.get_batch(frame_indices).asnumpy()
            print(f"[DEBUG sample] Successfully extracted {len(sampled_frames_array)} frames, shape: {sampled_frames_array.shape}")
            assert len(frame_indices) > 0, f"Generated empty frame_indices for interval {start_frame}-{end_frame}"
        except Exception as e:
            print(f"[ERROR] Failed to process video '{self.video_path}' between frames {start_frame}-{end_frame}.")
            print(
                f"[ERROR] Frame indices attempted: {frame_indices if 'frame_indices' in locals() else 'Not generated'}")
            print(f"[ERROR] The original exception was: {e}")
            import traceback
            traceback.print_exc()
            return "", []

        # Build prompt with frame indices
        prompt_parts = []
        for frame_idx in frame_indices:
            prompt_parts.append(f"frame {frame_idx}: <image>")

        prompt_segment = "\n".join(prompt_parts)
        sampled_images = [Image.fromarray(frame) for frame in sampled_frames_array]
        print(f"[DEBUG sample] Converted to {len(sampled_images)} PIL Images")
        assert len(sampled_images) > 0, self.vr
        return prompt_segment, sampled_images
