import os
import sys
import cv2
import json
import re
import gc
import numpy as np
import requests
import base64
import decord
import time
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("CLIP_BACKEND", "clip")

from clip_module import get_clip_model

if os.environ.get("CLIP_BACKEND", "clip").lower() == "siglip":
    clip = None
else:
    import clip

PROMPT_DIR = Path(__file__).parent / "prompt"

with open(PROMPT_DIR / "system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read().strip()

with open(PROMPT_DIR / "user_prompt.txt", "r") as f:
    USER_PROMPT_TEMPLATE = f.read().strip()

with open(PROMPT_DIR / "turn_prompt.txt", "r") as f:
    TURN_PROMPT = f.read().strip()

CONFIG = {
    "API_URL": "",
    "MODEL_NAME": "",
    "TARGET_JSON_PATH": "",
    "BASE_VIDEO_DIR": "",
    "OUTPUT_JSON_PATH": "",
    "NUM_INITIAL_FRAMES": 16,
    "CLIP_SAMPLE_FRAMES": 4,
    "UNIFORM_SAMPLE_FRAMES": 8,
    "MAX_FRAME_WIDTH": 640,
    "MAX_FRAME_HEIGHT": 360,
    "CLIP_GPU_ID": 0,
    "CLIP_MIN_FREE_GB": 2.0,
    "REQUEST_TIMEOUT": 90,
    "MAX_RETRIES": 6,
    "CLIP_BACKEND": "siglip",
}

os.environ["CLIP_BACKEND"] = CONFIG.get("CLIP_BACKEND", "clip").lower()

_CLIP_MODEL = None

def get_clip_device() -> str:
    try:
        if not torch.cuda.is_available():
            return "cpu"
        gpu_id = CONFIG["CLIP_GPU_ID"]
        min_free_gb = CONFIG["CLIP_MIN_FREE_GB"]
        free_mem = torch.cuda.mem_get_info(gpu_id)[0]
        free_gb = free_mem / (1024 ** 3)
        if free_gb >= min_free_gb:
            return f"cuda:{gpu_id}"
        else:
            return "cpu"
    except:
        return "cpu"

def get_clip_model_cached():
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        device = get_clip_device()
        _CLIP_MODEL = get_clip_model(device)
    return _CLIP_MODEL

def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def get_video_path(video_id: str) -> Optional[str]:
    base_dir = CONFIG["BASE_VIDEO_DIR"]
    for ext in [".mp4", ".webm", ".mkv"]:
        path = os.path.join(base_dir, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    return None

def get_video_metadata(video_path: str) -> Tuple[float, int]:
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        frame_count = len(vr)
        fps = vr.get_avg_fps()
        del vr
        return fps, frame_count
    except:
        return 0, 0

def scale_frame(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame
    scale = min(max_width / w, max_height / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def extract_frames_uniform(video_path: str, num_frames: int, max_width: int, max_height: int,
                           start_frame: int = None, end_frame: int = None) -> List[Tuple[int, np.ndarray]]:
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frames = len(vr)
    except:
        return []

    start_frame = max(0, start_frame or 0)
    end_frame = min(total_frames, end_frame or total_frames)

    indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        try:
            frame_rgb = vr[idx].asnumpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frames.append((idx, scale_frame(frame_bgr, max_width, max_height)))
            del frame_rgb, frame_bgr
        except:
            continue
    del vr
    gc.collect()
    return frames

def clip_sample_frames(video_path: str, start_frame: int, end_frame: int, num_frames: int,
                       prompt: str, max_width: int, max_height: int) -> List[Tuple[int, np.ndarray]]:
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frames = len(vr)
    except:
        return []

    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    frame_range = end_frame - start_frame

    candidate_count = min(frame_range, 256 if frame_range > 20000 else 128)

    if frame_range <= candidate_count:
        candidate_indices = list(range(start_frame, end_frame))
    else:
        candidate_indices = sorted(set(np.linspace(start_frame, end_frame - 1, candidate_count, dtype=int).tolist()))

    try:
        model, preprocess, tokenizer, inference_lock, device_obj = get_clip_model_cached()

        text_tokens = tokenizer([prompt]).to(device_obj)
        with torch.no_grad():
            text_feat = model.encode_text(text_tokens).cpu().numpy().squeeze()
        text_feat = text_feat / np.linalg.norm(text_feat)
        del text_tokens
    except Exception as e:
        del vr
        return extract_frames_uniform(video_path, num_frames, max_width, max_height, start_frame, end_frame)

    scored = []
    for idx in candidate_indices:
        try:
            frame_rgb = vr[idx].asnumpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_scaled = scale_frame(frame_bgr, max_width, max_height)

            pil_img = Image.fromarray(cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2RGB))
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device_obj)
            del pil_img

            with torch.no_grad():
                img_feat = model.encode_image(img_tensor).cpu().numpy().squeeze()
            img_feat = img_feat / np.linalg.norm(img_feat)
            del img_tensor

            sim = float(np.dot(img_feat, text_feat))
            scored.append((idx, sim, frame_scaled))

            del frame_rgb, frame_bgr, img_feat
        except:
            continue

    del vr

    if not scored:
        gc.collect()
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = sorted(scored[:num_frames], key=lambda x: x[0])

    result = [(idx, frame_bgr) for idx, _, frame_bgr in top_k]

    del scored, text_feat
    gc.collect()

    return result

def call_api(messages: List[Dict], max_retries: int = 5, timeout: int = 300) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            api_messages = []
            for msg in messages:
                if msg["role"] in ["system", "assistant"]:
                    api_messages.append(msg)
                elif msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, str):
                        api_messages.append({"role": "user", "content": content})
                    elif isinstance(content, list):
                        api_content = []
                        for item in content:
                            if item["type"] == "text":
                                api_content.append({"type": "text", "text": item["text"]})
                            elif item["type"] == "image":
                                api_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(item['image'])}"}
                                })
                        api_messages.append({"role": "user", "content": api_content})

            payload = {
                "model": CONFIG["MODEL_NAME"],
                "messages": api_messages,
                "max_tokens": 32678,
                "temperature": 0,
            }

            resp = requests.post(CONFIG["API_URL"], json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
    return None

def parse_response(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    think = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
    tool = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return (
        think.group(1).strip() if think else None,
        tool.group(1).strip() if tool else None,
        answer.group(1).strip() if answer else None
    )

def process_single_task(qa_item: Dict) -> Dict:
    video_id = qa_item.get('video', '')
    question = qa_item.get('question', '')
    video_path = get_video_path(video_id)

    result = qa_item.copy()

    if not video_path or not os.path.exists(video_path):
        result["answer"] = None
        result["error"] = "video_not_found"
        return result

    fps, frame_count = get_video_metadata(video_path)
    if frame_count == 0:
        result["answer"] = None
        result["error"] = "video_error"
        return result

    max_w, max_h = CONFIG["MAX_FRAME_WIDTH"], CONFIG["MAX_FRAME_HEIGHT"]

    initial_frames = extract_frames_uniform(video_path, CONFIG["NUM_INITIAL_FRAMES"], max_w, max_h)
    if not initial_frames:
        result["answer"] = None
        result["error"] = "frame_extraction_error"
        return result

    user_prompt = USER_PROMPT_TEMPLATE.format(question, frame_count, fps)
    user_content = [{"type": "text", "text": user_prompt}]
    for frame_idx, frame_bgr in initial_frames:
        user_content.append({"type": "text", "text": f"\nframe {frame_idx}: "})
        user_content.append({"type": "image", "image": frame_bgr})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    final_answer = None

    while True:
        response = call_api(messages, CONFIG["MAX_RETRIES"], CONFIG["REQUEST_TIMEOUT"])

        if not response:
            break

        think, tool_call, answer = parse_response(response)

        if answer:
            final_answer = answer
            break

        if tool_call:
            try:
                tool_data = json.loads(tool_call)
                name = tool_data.get("name")
                args = tool_data.get("arguments", {})

                if name == "clip_sample":
                    frames = clip_sample_frames(
                        video_path, int(args["start_frame"]), int(args["end_frame"]),
                        CONFIG["CLIP_SAMPLE_FRAMES"], args.get("prompt", question),
                        max_w, max_h
                    )
                elif name == "uniform_sample":
                    frames = extract_frames_uniform(
                        video_path, CONFIG["UNIFORM_SAMPLE_FRAMES"], max_w, max_h,
                        int(args["start_frame"]), int(args["end_frame"])
                    )
                else:
                    break

                if not frames:
                    break

                messages.append({"role": "assistant", "content": response})

                turn_content = [{"type": "text", "text": TURN_PROMPT}]
                for frame_idx, frame_bgr in frames:
                    turn_content.append({"type": "text", "text": f"\nframe {frame_idx}: "})
                    turn_content.append({"type": "image", "image": frame_bgr})
                messages.append({"role": "user", "content": turn_content})

            except Exception as e:
                break
        else:
            break

    result["answer"] = final_answer
    return result

def main():
    with open(CONFIG["TARGET_JSON_PATH"], 'r') as f:
        dataset = json.load(f)


    results = []
    for i, qa_item in enumerate(dataset):
        result = process_single_task(qa_item)
        results.append(result)

    with open(CONFIG["OUTPUT_JSON_PATH"], 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    answered = sum(1 for r in results if r.get("answer"))

if __name__ == "__main__":
    main()
