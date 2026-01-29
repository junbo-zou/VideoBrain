
import torch
import threading
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys
import os

try:
    from .clip_similarity import CLIPSimilarityCalculator
except ImportError as e:
    raise

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMemoryMonitor:

    def __init__(self):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            self.visible_gpus = [int(d) for d in visible_devices if d.strip()]
        else:
            self.visible_gpus = list(range(torch.cuda.device_count()))

        self.n_gpus = torch.cuda.device_count()
        self.enabled = self.n_gpus > 0
        self.gpu_count = self.n_gpus

        self.pynvml_initialized = False
        if PYNVML_AVAILABLE and self.enabled:
            try:
                pynvml.nvmlInit()
                self.pynvml_initialized = True
            except Exception as e:
                self.pynvml_initialized = False

    def get_gpu_memory_info(self, gpu_id: int) -> Tuple[float, int, int]:
        if gpu_id >= self.n_gpus or gpu_id < 0:
            return 1.0, 0, 0

        if not self.enabled:
            return 0.5, 0, 0

        if self.pynvml_initialized:
            try:
                physical_gpu_id = self.visible_gpus[gpu_id] if self.visible_gpus else gpu_id
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                usage_ratio = info.used / info.total if info.total > 0 else 0.5
                return usage_ratio, info.free, info.total
            except Exception as e:
                pass

        try:
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id)
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            usage_ratio = allocated / total if total > 0 else 0.5
            free = total - allocated
            return usage_ratio, free, total
        except Exception as e:
            return 0.5, 0, 0

    def get_gpu_memory_usage(self, gpu_id: int) -> float:
        usage_ratio, _, _ = self.get_gpu_memory_info(gpu_id)
        return usage_ratio

    def find_available_gpu(self, threshold: float = 0.90, min_free_gb: float = 2.0) -> Optional[int]:
        if self.gpu_count == 0:
            return None

        min_free_bytes = int(min_free_gb * 1024**3)
        best_gpu = None
        best_usage = 1.0

        for gpu_id in range(self.gpu_count):
            usage_ratio, free_bytes, total_bytes = self.get_gpu_memory_info(gpu_id)
            if usage_ratio < threshold and free_bytes >= min_free_bytes and usage_ratio < best_usage:
                best_gpu = gpu_id
                best_usage = usage_ratio

        return best_gpu

    def __del__(self):
        if self.pynvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class CLIPModelPool:

    def __init__(self, model_path: Optional[str] = None, preload: bool = True):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.lock = threading.Lock()
        self.monitor = GPUMemoryMonitor()


        if preload:
            self._init_model()

    def _init_model(self):

        if self.monitor.gpu_count > 0:
            gpu_id = 0
            usage_ratio, free_bytes, total_bytes = self.monitor.get_gpu_memory_info(gpu_id)
            free_gb = free_bytes / (1024**3)


            if usage_ratio < 0.90 and free_bytes >= 2 * 1024**3:
                try:
                    self.device = "cuda:0"
                    self.model = CLIPSimilarityCalculator(self.model_path, self.device)
                    return
                except Exception as e:
                    pass

        self.device = "cpu"
        self.model = CLIPSimilarityCalculator(self.model_path, self.device)

    def get_model(self) -> Tuple[CLIPSimilarityCalculator, str, threading.Lock]:
        if self.model is None:
            with self.lock:
                if self.model is None:
                    self._init_model()

        return self.model, self.device, self.lock

    def get_status(self) -> Dict:
        status = {
            "gpu_count": self.monitor.n_gpus,
            "visible_gpus": self.monitor.visible_gpus if hasattr(self.monitor, 'visible_gpus') else list(range(self.monitor.n_gpus)),
            "device": self.device,
            "model_loaded": self.model is not None
        }

        if self.monitor.n_gpus > 0:
            usage_ratio, free_bytes, total_bytes = self.monitor.get_gpu_memory_info(0)
            status["gpu_0_memory"] = {
                "usage": f"{usage_ratio:.1%}",
                "free_gb": f"{free_bytes / (1024**3):.2f}",
                "total_gb": f"{total_bytes / (1024**3):.2f}"
            }

        return status


_clip_manager = None


def get_clip_manager(model_path: Optional[str] = None, preload: bool = True) -> CLIPModelPool:
    global _clip_manager
    if _clip_manager is None:
        _clip_manager = CLIPModelPool(model_path=model_path, preload=preload)
    return _clip_manager


def compute_clip_similarity_batch(
    frames: List[np.ndarray],
    prompt: str,
    batch_size: int = 8
) -> Tuple[List[float], List[int]]:
    from PIL import Image

    clip_manager = get_clip_manager()

    model, device, lock = clip_manager.get_model()

    try:
        text_features = model.encode_text(prompt)

        similarities = []

        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]


            for frame in batch_frames:
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)

                pil_image = Image.fromarray(frame)

                image_features = model.encode_image_pil(pil_image)

                similarity = float(np.dot(image_features, text_features))
                similarities.append(similarity)

        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

        return similarities, sorted_indices

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [0.5] * len(frames), list(range(len(frames)))