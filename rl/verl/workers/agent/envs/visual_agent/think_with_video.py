import re
import random
import time
import requests
import numpy as np
import requests
import base64
import json
import decord
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional
from time import sleep
from PIL import Image
from io import BytesIO
from math import ceil, floor
from PIL import Image
from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents
import cv2
import threading


class ThinkWithVideo(ToolBase):
    _vr_cache = {}  # key: (video_path, width, height, ctx_type), value: {'reader': VideoReader, 'last_used': timestamp}
    _cache_lock = threading.Lock()
    _max_cache_size = 5

    @staticmethod
    def _get_gpu_memory_gb(gpu_id=0):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = info.free / 1024**3
            pynvml.nvmlShutdown()
            return free_gb
        except Exception as e:
            print(f"[WARN] Failed to get GPU memory: {e}, defaulting to CPU")
            return 0.0

    @staticmethod
    def _get_video_context(gpu_id=0, min_free_gb=5.0, use_gpu=False):
        """Select decode context: GPU if use_gpu=True and memory sufficient, else CPU."""
        if not use_gpu:
            print(f"[INFO] Using CPU decode (default, better for random access)")
            return decord.cpu(0), 'cpu'

        free_gb = ThinkWithVideo._get_gpu_memory_gb(gpu_id)
        if free_gb >= min_free_gb:
            print(f"[INFO] GPU {gpu_id} has {free_gb:.2f}GB free, using GPU decode")
            return decord.gpu(gpu_id), 'gpu'
        else:
            print(f"[WARN] GPU {gpu_id} only has {free_gb:.2f}GB free (< {min_free_gb}GB), using CPU decode")
            return decord.cpu(0), 'cpu'

    @classmethod
    def _get_or_create_video_reader(cls, video_path, width, height, ctx, ctx_type):
        """Get or create a VideoReader with LRU caching."""
        import time
        cache_key = (video_path, width, height, ctx_type)

        with cls._cache_lock:
            if cache_key in cls._vr_cache:
                cache_entry = cls._vr_cache[cache_key]
                cache_entry['last_used'] = time.time()
                print(f"[INFO] Reusing cached VideoReader for {video_path} ({width}x{height}, {ctx_type})")
                return cache_entry['reader']

            if len(cls._vr_cache) >= cls._max_cache_size:
                oldest_key = min(cls._vr_cache.keys(),
                               key=lambda k: cls._vr_cache[k]['last_used'])
                removed_entry = cls._vr_cache.pop(oldest_key)
                del removed_entry
                import gc
                gc.collect()
                print(f"[INFO] Cache full, removed oldest: {oldest_key[0]} (forced GC, total: {len(cls._vr_cache)})")

            print(f"[INFO] Creating new VideoReader for {video_path} ({width}x{height}, {ctx_type})")
            vr = decord.VideoReader(
                video_path,
                ctx=ctx,
                num_threads=8,
                width=width,
                height=height
            )

            cls._vr_cache[cache_key] = {
                'reader': vr,
                'last_used': time.time()
            }
            print(f"[INFO] Cached VideoReader, total cached: {len(cls._vr_cache)}")
            return vr

    @classmethod
    def _clear_cache(cls):
        """Clear VideoReader cache."""
        with cls._cache_lock:
            cls._vr_cache.clear()
            print("[INFO] VideoReader cache cleared")
    name = "think_with_video"
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
        self.clip_preloaded = False  # Track if CLIP model has been preloaded

        self.num_frames_per_sample = None
        self.max_width = None
        self.max_height = None

        self.height = 0
        self.width = 0
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        self.turn_count += 1
        print(f"[DEBUG execute] Called with video_path={self.video_path}, total_frames={self.total_frames}, turn_count={self.turn_count}")

        if not self.video_path:
            print("[DEBUG execute] Terminating: no video_path")
            return '', 0.0, True, {}

        action_block = extract_tool_call_contents(self.action_start, self.action_end, action_string)
        if not action_block:
            print(f"[DEBUG execute] Terminating: no action_block found in action_string='{action_string}'")
            return '', 0.0, True, {}
        action_block = action_block[-1]
        print(f"[DEBUG execute] Extracted action_block: '{action_block}'")

        # Parse JSON
        try:
            action_data = json.loads(action_block.strip())
            print(f"[DEBUG execute] Parsed JSON action_data: {action_data}")
        except json.JSONDecodeError as e:
            print(f"[DEBUG execute] Terminating: JSON parsing failed with error: {e}")
            print(f"[DEBUG execute] Action block content: '{action_block}'")
            return '', 0.0, True, {}

        # ============ Format Detection ============
        # Detect format: old (action: "choose_frames") or new (name: "uniform_sample" / "clip_sample")

        if "name" in action_data:
            # NEW FORMAT: {"name": "uniform_sample" or "clip_sample", "arguments": {...}}
            tool_name = action_data.get("name")
            print(f"[DEBUG execute] Detected NEW format with tool_name='{tool_name}'")

            if tool_name == "uniform_sample":
                return self._execute_uniform_sample(action_data)
            elif tool_name == "clip_sample":
                return self._execute_clip_sample(action_data)
            else:
                print(f"[DEBUG execute] Terminating: unknown tool name '{tool_name}'")
                return '', 0.0, True, {}

        elif "action" in action_data:
            # OLD FORMAT: {"action": "choose_frames", "frame_range": [START, END]}
            print(f"[DEBUG execute] Detected OLD format with action='{action_data.get('action')}'")
            return self._execute_old_format(action_data)

        else:
            print(f"[DEBUG execute] Terminating: unknown format (no 'name' or 'action' field)")
            return '', 0.0, True, {}

    def _execute_old_format(self, action_data):
        """Execute old format: {"action": "choose_frames", "frame_range": [START, END]}"""
        # Check if action is "choose_frames"
        if action_data.get("action") != "choose_frames":
            print(f"[DEBUG execute] Terminating: unknown action '{action_data.get('action')}'")
            return '', 0.0, True, {}

        # Extract frame range
        frame_range = action_data.get("frame_range")
        if not frame_range or not isinstance(frame_range, list) or len(frame_range) != 2:
            print(f"[DEBUG execute] Terminating: invalid frame_range format: {frame_range}")
            return '', 0.0, True, {}

        start_frame = int(frame_range[0])
        end_frame = int(frame_range[1])
        print(f"[DEBUG execute] Parsed frames: start={start_frame}, end={end_frame}, total_frames={self.total_frames}, num_frames_per_sample={self.num_frames_per_sample}")

        if start_frame > self.total_frames:
            print(f"[DEBUG execute] Terminating: start_frame({start_frame}) > total_frames({self.total_frames})")
            return '', 0.0, True, {}
        if end_frame > self.total_frames:
            print(f"[DEBUG execute] Adjusting end_frame from {end_frame} to {self.total_frames}")
            end_frame = self.total_frames

        if start_frame >= end_frame - self.num_frames_per_sample:
            print(f"[DEBUG execute] Terminating: frame range too small, {start_frame} >= {end_frame - self.num_frames_per_sample}")
            return '', 0.0, True, {}

        print(f"[DEBUG execute] Calling _process_zoom_request_no_desc with start={start_frame}, end={end_frame}")
        user_msg, focused_images_data = self._process_zoom_request_no_desc(
            start_frame=start_frame,
            end_frame=end_frame
        )
        print(f"[DEBUG execute] _process_zoom_request_no_desc returned: msg_len={len(user_msg)}, num_images={len(focused_images_data)}")

        if len(focused_images_data) == 0:
            print("[DEBUG execute] Terminating: no images extracted (empty focused_images_data)")
            return '', 0.0, True, {}

        # Add instruction prefix before frame info
        instruction_prefix = self._build_instruction_prefix()
        full_user_msg = instruction_prefix + "\n" + user_msg
        all_user_msg = self.chat_template.format(full_user_msg)

        print(f"[DEBUG execute] Success! Returning {len(focused_images_data)} images with done=False")
        obs_dict = {
            "prompt": all_user_msg,
            "multi_modal_data": {
                "image": focused_images_data
            }
        }
        return obs_dict, 0.0, False, {}

    def _execute_uniform_sample(self, action_data):
        """Execute uniform_sample format: {"name": "uniform_sample", "arguments": {"start_frame": X, "end_frame": Y}}"""
        # Fixed number of frames for uniform sampling
        num_frame = 8

        # Extract arguments
        arguments = action_data.get("arguments")
        if not arguments or not isinstance(arguments, dict):
            print(f"[DEBUG execute] Terminating: invalid arguments format: {arguments}")
            return '', 0.0, True, {}

        start_frame = arguments.get("start_frame")
        end_frame = arguments.get("end_frame")

        # Validate parameters
        if start_frame is None or end_frame is None:
            print(f"[DEBUG execute] Terminating: missing required parameters (start_frame or end_frame)")
            return '', 0.0, True, {}

        try:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        except (ValueError, TypeError) as e:
            print(f"[DEBUG execute] Terminating: parameter type conversion failed: {e}")
            return '', 0.0, True, {}

        print(f"[DEBUG execute] Parsed parameters: start_frame={start_frame}, end_frame={end_frame}, num_frame={num_frame} (fixed)")

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

        if len(sampled_images) == 0:
            print("[DEBUG execute] Terminating: no images extracted (empty sampled_images)")
            return '', 0.0, True, {}

        # Add instruction prefix before frame info
        instruction_prefix = self._build_instruction_prefix()
        full_user_msg = instruction_prefix + "\n" + user_msg
        all_user_msg = self.chat_template.format(full_user_msg)

        print(f"[DEBUG execute] Success! Returning {len(sampled_images)} images with done=False")
        obs_dict = {
            "prompt": all_user_msg,
            "multi_modal_data": {
                "image": sampled_images
            }
        }
        return obs_dict, 0.0, False, {}

    def _execute_clip_sample(self, action_data):
        """Execute clip_sample format: {"name": "clip_sample", "arguments": {"start_frame": X, "end_frame": Y, "prompt": "..."}}"""
        # Fixed number of frames for clip sampling
        num_frame = 4

        # Extract arguments
        arguments = action_data.get("arguments")
        if not arguments or not isinstance(arguments, dict):
            print(f"[DEBUG execute] Terminating: invalid arguments format: {arguments}")
            return '', 0.0, True, {}

        start_frame = arguments.get("start_frame")
        end_frame = arguments.get("end_frame")
        prompt = arguments.get("prompt")

        # Validate parameters
        if start_frame is None or end_frame is None or prompt is None:
            print(f"[DEBUG execute] Terminating: missing required parameters (start_frame, end_frame, or prompt)")
            return '', 0.0, True, {}

        try:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            prompt = str(prompt).strip()
        except (ValueError, TypeError) as e:
            print(f"[DEBUG execute] Terminating: parameter type conversion failed: {e}")
            return '', 0.0, True, {}

        print(f"[DEBUG execute] Parsed parameters: start_frame={start_frame}, end_frame={end_frame}, num_frame={num_frame} (fixed), prompt='{prompt}'")

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

        if len(sampled_images) == 0:
            print("[DEBUG execute] Terminating: no images extracted (empty sampled_images)")
            return '', 0.0, True, {}

        # Add instruction prefix before frame info
        instruction_prefix = self._build_instruction_prefix()
        full_user_msg = instruction_prefix + "\n" + user_msg
        all_user_msg = self.chat_template.format(full_user_msg)

        print(f"[DEBUG execute] Success! Returning {len(sampled_images)} images with done=False")
        obs_dict = {
            "prompt": all_user_msg,
            "multi_modal_data": {
                "image": sampled_images
            }
        }
        return obs_dict, 0.0, False, {}

    def _build_instruction_prefix(self):
        if self.turn_count >= 5:
            return """Start with <thinking>. Format strictly as: <thinking>...</thinking><answer>...</answer>

Based on sampling, here are the following frames

You have already called the tool 5 times. Further tool calls are prohibited in this round. Please provide the answer to this question in this round."""
        else:
            return """Start with <thinking>. Format strictly as: <thinking>...</thinking><tool_call>...</tool_call> or <thinking>...</thinking><answer>...</answer>

Based on sampling, here are the following frames

If you need to call the "uniform_sample" function again in this round, please update your range.
If you need to call the "clip_sample" function again in this round, please refine your prompt for better retrieval."""

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.video_path = kwargs.get('video_path')
        self.fps = kwargs.get('fps')
        self.total_frames = kwargs.get('total_frames')
        self.height = kwargs.get('height')
        self.width = kwargs.get('width')
        self.question = kwargs.get('question', '')
        self.vr = None
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        self.turn_count = 0

        print(f"[DEBUG reset] Initializing ThinkWithVideo:")
        print(f"  video_path: {self.video_path}")
        print(f"  fps: {self.fps}")
        print(f"  total_frames: {self.total_frames}")
        print(f"  height: {self.height}, width: {self.width}")
        print(f"  question: {self.question[:50]}..." if len(self.question) > 50 else f"  question: {self.question}")

        if self.total_frames and self.fps:
            duration_seconds = self.total_frames / self.fps
            print(f"  duration_seconds: {duration_seconds:.2f}")
            if duration_seconds > 300:
                self.num_frames_per_sample = 8
                self.max_width = 640
                self.max_height = 360
            else:
                self.num_frames_per_sample = 8
                self.max_width = 640
                self.max_height = 360
        else:
            self.num_frames_per_sample = 8
            self.max_width = 640
            self.max_height = 360

        print(f"  num_frames_per_sample: {self.num_frames_per_sample}")
        print(f"  max_width: {self.max_width}, max_height: {self.max_height}")

        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width
        print(f"  Initial image dimensions from multi_modal_data: {self.width}x{self.height}")

        # Preload CLIP model to avoid 40-second delay on first clip_sample call
        if not self.clip_preloaded:
            print(f"[DEBUG reset] Preloading CLIP model (first time)...")
            import time
            start_time = time.time()
            from verl.workers.agent.envs.visual_agent.clip_module.clip_manager import get_clip_manager
            self.clip_manager = get_clip_manager(preload=True)
            model, device, lock = self.clip_manager.get_model()
            elapsed = time.time() - start_time
            print(f"[DEBUG reset] ✓ CLIP model preloaded on {device} in {elapsed:.2f}s")
            self.clip_preloaded = True
        else:
            print(f"[DEBUG reset] CLIP model already preloaded, skipping")

        print(f"[DEBUG reset] Reset complete!")

    def _process_zoom_request_no_desc(
            self,
            start_frame: int,
            end_frame: int
    ) -> Tuple[str, List[np.ndarray]]:
        sample_start = start_frame + 1
        sample_end = end_frame - 1
        print(f"[DEBUG zoom] Processing frames: sample_start={sample_start}, sample_end={sample_end}")

        try:
            if self.vr is None:
                print(f"[DEBUG zoom] Initializing VideoReader for path: {self.video_path}")
                max_retries = 3
                base_delay = 2
                for attempt in range(max_retries):
                    try:
                        target_w, target_h = self._calculate_target_dims()
                        print(f"[DEBUG zoom] Target dimensions: {target_w}x{target_h}")

                        # Select GPU/CPU decode context
                        ctx, ctx_type = self._get_video_context(gpu_id=0, min_free_gb=5.0)

                        # Get VideoReader from cache
                        self.vr = self._get_or_create_video_reader(
                            self.video_path,
                            target_w,
                            target_h,
                            ctx,
                            ctx_type
                        )
                        print(f"[DEBUG zoom] VideoReader initialized successfully, total frames in video: {len(self.vr)}")
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

            frame_indices = sorted(
                list(set(map(int, np.linspace(sample_start, sample_end, self.num_frames_per_sample)))))
            print(f"[DEBUG zoom] Generated frame_indices: {frame_indices}")

            focused_frames_array = self.vr.get_batch(frame_indices).asnumpy()
            print(f"[DEBUG zoom] Successfully extracted {len(focused_frames_array)} frames, shape: {focused_frames_array.shape}")
            assert len(frame_indices) > 0, f"Generated empty frame_indices for interval {sample_start}-{sample_end}"
        except Exception as e:
            print(f"[ERROR] Failed to process video '{self.video_path}' between frames {start_frame}-{end_frame}.")
            print(
                f"[ERROR] Frame indices attempted: {frame_indices if 'frame_indices' in locals() else 'Not generated'}")
            print(f"[ERROR] The original exception was: {e}")
            import traceback
            traceback.print_exc()
            return "", []

        prompt_parts = []
        for frame_idx in frame_indices:
            prompt_parts.append(f"frame {frame_idx}: <image>")

        focused_prompt_segment = "\n".join(prompt_parts)
        focused_images_data = [Image.fromarray(frame) for frame in focused_frames_array]
        print(f"[DEBUG zoom] Converted to {len(focused_images_data)} PIL Images")
        assert len(focused_images_data) > 0, self.vr
        return focused_prompt_segment, focused_images_data

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
            target_w, target_h = 640, 360
        else:
            target_w, target_h = 640, 360

        print(f"[DEBUG uniform_sample] Processing frames: start={start_frame}, end={end_frame}, num={num_frame}")
        print(f"[DEBUG uniform_sample] Target dimensions: {target_w}x{target_h}")

        try:
            # Select GPU/CPU decode context
            ctx, ctx_type = self._get_video_context(gpu_id=0, min_free_gb=5.0)

            # Get VideoReader from cache
            self.vr = self._get_or_create_video_reader(
                self.video_path,
                target_w,
                target_h,
                ctx,
                ctx_type
            )
            print(f"[DEBUG uniform_sample] VideoReader ready, total frames in video: {len(self.vr)}")

            # Generate uniformly spaced frame indices
            # Use linspace to get evenly distributed indices between start_frame and end_frame
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frame, dtype=int).tolist()
            # Remove duplicates and sort
            frame_indices = sorted(list(set(frame_indices)))
            print(f"[DEBUG uniform_sample] Generated frame_indices: {frame_indices}")

            sampled_frames_array = self.vr.get_batch(frame_indices).asnumpy()
            print(f"[DEBUG uniform_sample] Successfully extracted {len(sampled_frames_array)} frames, shape: {sampled_frames_array.shape}")
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
        print(f"[DEBUG uniform_sample] Converted to {len(sampled_images)} PIL Images")
        assert len(sampled_images) > 0, self.vr
        return prompt_segment, sampled_images

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
        if hasattr(self, 'clip_manager') and self.clip_manager:
            print(f"[DEBUG clip_sample] Using preloaded CLIP model")
            from verl.workers.agent.envs.visual_agent.clip_module.clip_manager import compute_clip_similarity_batch
        else:
            print(f"[DEBUG clip_sample] First time loading CLIP model, initializing...")
            from verl.workers.agent.envs.visual_agent.clip_module.clip_manager import compute_clip_similarity_batch
            from verl.workers.agent.envs.visual_agent.clip_module.clip_manager import get_clip_manager
            self.clip_manager = get_clip_manager(preload=True)
            self.clip_preloaded = True

        # Fixed resolution for CLIP
        target_w, target_h = 640, 360
        frame_range = end_frame - start_frame

        # Determine sampling count based on frame range
        if frame_range < 20000:
            sample_count = min(128, frame_range)
        else:
            sample_count = min(256, frame_range)

        print(f"[DEBUG clip_sample] Frame range: {frame_range}, sampling {sample_count} frames for CLIP")
        print(f"[DEBUG clip_sample] Target dimensions: {target_w}x{target_h}")

        try:
            # Select GPU/CPU decode context
            ctx, ctx_type = self._get_video_context(gpu_id=0, min_free_gb=5.0)

            # Get VideoReader from cache
            self.vr = self._get_or_create_video_reader(
                self.video_path,
                target_w,
                target_h,
                ctx,
                ctx_type
            )
            print(f"[DEBUG clip_sample] VideoReader ready, total frames in video: {len(self.vr)}")

            # Generate uniformly spaced frame indices for CLIP processing
            clip_frame_indices = np.linspace(start_frame, end_frame - 1, sample_count, dtype=int).tolist()
            # Remove duplicates and sort
            clip_frame_indices = sorted(list(set(clip_frame_indices)))
            print(f"[DEBUG clip_sample] Generated {len(clip_frame_indices)} frame indices for CLIP: {clip_frame_indices[:10]}...")

            # Extract frames
            sampled_frames_array = self.vr.get_batch(clip_frame_indices).asnumpy()
            print(f"[DEBUG clip_sample] Successfully extracted {len(sampled_frames_array)} frames, shape: {sampled_frames_array.shape}")

            # Compute CLIP similarity (batch_size=32 for better GPU utilization)
            print(f"[DEBUG clip_sample] Computing CLIP similarity with prompt: '{prompt}'")
            similarities, sorted_indices = compute_clip_similarity_batch(
                frames=sampled_frames_array,
                prompt=prompt,
                batch_size=32
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

            # Build prompt with frame indices (without similarity scores)
            prompt_parts = []
            for frame_idx in final_frame_indices:
                prompt_parts.append(f"frame {frame_idx}: <image>")

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

    def _calculate_target_dims(self) -> tuple[int, int]:
        w, h = self.width, self.height

        if w <= self.max_width and h <= self.max_height:
            return w, h

        ratio_w = self.max_width / w
        ratio_h = self.max_height / h
        scale_ratio = min(ratio_w, ratio_h)

        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)

        return new_w, new_h