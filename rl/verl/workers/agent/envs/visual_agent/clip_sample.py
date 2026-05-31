import re
import random
import time
import json
import decord
import numpy as np
from typing import List, Dict, Tuple, Any
from PIL import Image
from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents


class ClipSample(ToolBase):
    name = "clip_sample"
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

        # Fixed resolution for CLIP
        self.target_w = 640
        self.target_h = 360

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

        # Parse JSON format: {"name": "clip_sample", "arguments": {"start_frame": X, "end_frame": Y, "num_frame": Z, "prompt": "text"}}
        try:
            action_data = json.loads(action_block.strip())
            print(f"[DEBUG execute] Parsed JSON action_data: {action_data}")
        except json.JSONDecodeError as e:
            print(f"[DEBUG execute] Terminating: JSON parsing failed with error: {e}")
            print(f"[DEBUG execute] Action block content: '{action_block}'")
            return '', 0.0, True, {}

        # Check if name is "clip_sample"
        if action_data.get("name") != "clip_sample":
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
        prompt = arguments.get("prompt")

        # Validate parameters
        if start_frame is None or end_frame is None or num_frame is None or prompt is None:
            print(f"[DEBUG execute] Terminating: missing required parameters")
            return '', 0.0, True, {}

        try:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            num_frame = int(num_frame)
            prompt = str(prompt).strip()
        except (ValueError, TypeError) as e:
            print(f"[DEBUG execute] Terminating: parameter type conversion failed: {e}")
            return '', 0.0, True, {}

        print(f"[DEBUG execute] Parsed parameters: start_frame={start_frame}, end_frame={end_frame}, num_frame={num_frame}, prompt='{prompt}'")

        # Check num_frame constraint (>20 is format error)
        if num_frame > 20:
            print(f"[DEBUG execute] Terminating: num_frame({num_frame}) > 20 (format error)")
            return '', 0.0, True, {}

        if not prompt:
            print(f"[DEBUG execute] Terminating: empty prompt")
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

        print(f"[DEBUG execute] Calling _clip_sample_frames with start={start_frame}, end={end_frame}, num={num_frame}, prompt='{prompt}'")
        user_msg, sampled_images = self._clip_sample_frames(
            start_frame=start_frame,
            end_frame=end_frame,
            num_frame=num_frame,
            prompt=prompt
        )
        print(f"[DEBUG execute] _clip_sample_frames returned: msg_len={len(user_msg)}, num_images={len(sampled_images)}")

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

        print(f"[DEBUG reset] Initializing ClipSample:")
        print(f"  video_path: {self.video_path}")
        print(f"  fps: {self.fps}")
        print(f"  total_frames: {self.total_frames}")
        print(f"  height: {self.height}, width: {self.width}")
        print(f"  Fixed CLIP resolution: {self.target_w}x{self.target_h}")

        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width
        print(f"  Initial image dimensions from multi_modal_data: {self.width}x{self.height}")
        print(f"[DEBUG reset] Reset complete!")

    def _clip_sample_frames(
            self,
            start_frame: int,
            end_frame: int,
            num_frame: int,
            prompt: str
    ) -> Tuple[str, List[Image.Image]]:
        """
        Use CLIP to sample frames based on similarity to prompt

        Sampling strategy:
        - Frame range < 20000: uniformly sample 128 frames
        - Frame range >= 20000: uniformly sample 256 frames
        - Return top num_frame frames with highest CLIP similarity
        - Re-sort selected frames by temporal order

        Fixed resolution: 448x252 for all frames
        """
        # Import here to avoid circular import
        from verl.workers.agent.envs.visual_agent.clip_module.clip_manager import compute_clip_similarity_batch

        frame_range = end_frame - start_frame

        # Determine sampling count based on frame range
        if frame_range < 20000:
            sample_count = min(128, frame_range)
        else:
            sample_count = min(256, frame_range)

        print(f"[DEBUG clip_sample] Frame range: {frame_range}, sampling {sample_count} frames for CLIP")

        try:
            if self.vr is None:
                print(f"[DEBUG clip_sample] Initializing VideoReader for path: {self.video_path}")
                max_retries = 3
                base_delay = 2
                for attempt in range(max_retries):
                    try:
                        self.vr = decord.VideoReader(
                            self.video_path,
                            ctx=decord.cpu(0),
                            width=self.target_w,
                            height=self.target_h
                        )
                        print(f"[DEBUG clip_sample] VideoReader initialized successfully, total frames in video: {len(self.vr)}")
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

            # Generate uniformly spaced frame indices for CLIP processing
            clip_frame_indices = np.linspace(start_frame, end_frame - 1, sample_count, dtype=int).tolist()
            # Remove duplicates and sort
            clip_frame_indices = sorted(list(set(clip_frame_indices)))
            print(f"[DEBUG clip_sample] Generated {len(clip_frame_indices)} frame indices for CLIP: {clip_frame_indices[:10]}...")

            # Extract frames
            sampled_frames_array = self.vr.get_batch(clip_frame_indices).asnumpy()
            print(f"[DEBUG clip_sample] Successfully extracted {len(sampled_frames_array)} frames, shape: {sampled_frames_array.shape}")

            # Compute CLIP similarity (batch_size=8)
            print(f"[DEBUG clip_sample] Computing CLIP similarity with prompt: '{prompt}'")
            similarities, sorted_indices = compute_clip_similarity_batch(
                frames=sampled_frames_array,
                prompt=prompt,
                batch_size=8
            )

            print(f"[DEBUG clip_sample] CLIP similarity computed")
            print(f"  Mean similarity: {np.mean(similarities):.4f}")
            print(f"  Max similarity: {np.max(similarities):.4f}")
            print(f"  Min similarity: {np.min(similarities):.4f}")

            # Select top num_frame frames by similarity
            top_indices = sorted_indices[:num_frame]
            print(f"[DEBUG clip_sample] Selected top {num_frame} frames by similarity")

            # Get corresponding frame indices and frames
            selected_frame_indices = [clip_frame_indices[i] for i in top_indices]
            selected_frames = [sampled_frames_array[i] for i in top_indices]
            selected_similarities = [similarities[i] for i in top_indices]

            # Re-sort by temporal order (frame index)
            temporal_order = sorted(range(len(selected_frame_indices)), key=lambda i: selected_frame_indices[i])
            final_frame_indices = [selected_frame_indices[i] for i in temporal_order]
            final_frames = [selected_frames[i] for i in temporal_order]
            final_similarities = [selected_similarities[i] for i in temporal_order]

            print(f"[DEBUG clip_sample] Final frame indices (temporal order): {final_frame_indices}")
            print(f"[DEBUG clip_sample] Corresponding similarities: {[f'{s:.4f}' for s in final_similarities]}")

            # Build prompt with frame indices and similarity scores
            prompt_parts = []
            for frame_idx, similarity in zip(final_frame_indices, final_similarities):
                prompt_parts.append(f"frame {frame_idx} (similarity: {similarity:.3f}): <image>")

            prompt_segment = "\n".join(prompt_parts)
            final_images = [Image.fromarray(frame) for frame in final_frames]
            print(f"[DEBUG clip_sample] Converted to {len(final_images)} PIL Images")

            assert len(final_images) > 0, "No frames selected by CLIP"
            return prompt_segment, final_images

        except Exception as e:
            print(f"[ERROR] Failed to process video '{self.video_path}' with CLIP sampling.")
            print(f"[ERROR] The original exception was: {e}")
            import traceback
            traceback.print_exc()
            return "", []
