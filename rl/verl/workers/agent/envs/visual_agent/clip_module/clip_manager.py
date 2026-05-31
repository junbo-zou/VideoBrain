"""
CLIP Manager for GPU allocation and model management
Manages CLIP models across multiple GPUs with load balancing
Fixed for Ray distributed workers
"""

import torch
import threading
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys
import os

# Import from local module
try:
    from .clip_similarity import CLIPSimilarityCalculator
except ImportError as e:
    print(f"Error importing clip_similarity: {e}")
    raise

# Import nvidia-ml-py (pynvml) for accurate GPU memory detection
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[CLIPManager] Warning: nvidia-ml-py (pynvml) not available, falling back to torch.cuda for memory detection")
    print("[CLIPManager] Install with: pip install nvidia-ml-py")


class GPUMemoryMonitor:
    """Monitor GPU memory usage using pynvml for accurate detection"""

    def __init__(self):
        """Initialize GPU monitoring"""
        # Get GPUs visible to this process
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            self.visible_gpus = [int(d) for d in visible_devices if d.strip()]
            print(f"[GPUMemoryMonitor] CUDA_VISIBLE_DEVICES: {visible_devices}")
        else:
            self.visible_gpus = list(range(torch.cuda.device_count()))

        # Actual GPU count available to this process
        self.n_gpus = torch.cuda.device_count()
        print(f"[GPUMemoryMonitor] Process has access to {self.n_gpus} GPU(s): {self.visible_gpus if self.visible_gpus else 'all'}")
        self.enabled = self.n_gpus > 0
        self.gpu_count = self.n_gpus

        # Initialize pynvml if available
        self.pynvml_initialized = False
        if PYNVML_AVAILABLE and self.enabled:
            try:
                pynvml.nvmlInit()
                self.pynvml_initialized = True
                print(f"[GPUMemoryMonitor] pynvml initialized successfully")
            except Exception as e:
                print(f"[GPUMemoryMonitor] Warning: Failed to initialize pynvml: {e}")
                self.pynvml_initialized = False

    def get_gpu_memory_info(self, gpu_id: int) -> Tuple[float, int, int]:
        """
        Get GPU memory usage using pynvml (accurate, includes vLLM memory)

        Args:
            gpu_id: GPU device ID (local to this process)

        Returns:
            Tuple of (usage_ratio, free_bytes, total_bytes)
        """
        # Check if GPU ID is valid for this process
        if gpu_id >= self.n_gpus or gpu_id < 0:
            return 1.0, 0, 0  # Return full for non-existent GPUs

        if not self.enabled:
            return 0.5, 0, 0

        # Try pynvml first (accurate, includes vLLM memory)
        if self.pynvml_initialized:
            try:
                # Map local GPU ID to physical GPU ID
                physical_gpu_id = self.visible_gpus[gpu_id] if self.visible_gpus else gpu_id
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                usage_ratio = info.used / info.total if info.total > 0 else 0.5
                return usage_ratio, info.free, info.total
            except Exception as e:
                print(f"[GPUMemoryMonitor] Warning: pynvml failed for GPU {gpu_id}: {e}")

        # Fallback to torch.cuda (less accurate, may not see vLLM memory)
        try:
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id)
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            usage_ratio = allocated / total if total > 0 else 0.5
            free = total - allocated
            return usage_ratio, free, total
        except Exception as e:
            print(f"[GPUMemoryMonitor] Warning: Failed to get GPU {gpu_id} memory: {e}")
            return 0.5, 0, 0

    def get_gpu_memory_usage(self, gpu_id: int) -> float:
        """
        Get GPU memory usage percentage

        Args:
            gpu_id: GPU device ID (local to this process)

        Returns:
            Memory usage percentage (0.0 to 1.0)
        """
        usage_ratio, _, _ = self.get_gpu_memory_info(gpu_id)
        return usage_ratio

    def find_available_gpu(self, threshold: float = 0.90, min_free_gb: float = 2.0) -> Optional[int]:
        """
        Find a GPU with sufficient free memory (and optionally below usage threshold)

        Args:
            threshold: Maximum memory usage threshold (set to 1.0 to ignore usage rate)
            min_free_gb: Minimum free memory in GB (default 2.0 GB)

        Returns:
            GPU ID with most free memory that meets requirements, or None if none qualify
        """
        if self.gpu_count == 0:
            return None

        min_free_bytes = int(min_free_gb * 1024**3)  # Convert GB to bytes
        best_gpu = None
        max_free_bytes = 0  # Track GPU with most free memory

        for gpu_id in range(self.gpu_count):
            usage_ratio, free_bytes, total_bytes = self.get_gpu_memory_info(gpu_id)
            # Check if GPU meets requirements
            if free_bytes >= min_free_bytes and usage_ratio < threshold:
                # Select GPU with most free memory
                if free_bytes > max_free_bytes:
                    best_gpu = gpu_id
                    max_free_bytes = free_bytes

        if best_gpu is not None:
            _, free_bytes, _ = self.get_gpu_memory_info(best_gpu)
            print(f"[GPUMemoryMonitor] Selected GPU {best_gpu} with {free_bytes/(1024**3):.2f}GB free")

        return best_gpu

    def __del__(self):
        """Clean up resources"""
        if self.pynvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class CLIPModelPool:
    """
    Simplified CLIP model manager for Ray distributed workers
    Each worker uses its local GPU 0, with CPU fallback
    """

    def __init__(self, model_path: Optional[str] = None, preload: bool = True):
        """
        Initialize CLIP model pool

        Args:
            model_path: Path to CLIP model (optional)
            preload: Whether to preload model on local GPU
        """
        self.model_path = model_path
        self.model = None
        self.device = None
        self.lock = threading.Lock()
        self.monitor = GPUMemoryMonitor()

        print(f"[CLIPModelPool] Initializing with {self.monitor.gpu_count} GPU(s) available to this process")

        # Preload model if requested
        if preload:
            self._init_model()

    def _init_model(self):
        """Initialize CLIP model with better GPU distribution"""
        print("[CLIPModelPool] Initializing CLIP model...")

        # Try to distribute across GPUs using process ID or find least loaded GPU
        if self.monitor.gpu_count > 0:
            # Method 1: Use process ID for round-robin distribution
            import os
            pid = os.getpid()
            gpu_id = pid % self.monitor.gpu_count
            print(f"[CLIPModelPool] Process {pid} attempting to use GPU {gpu_id} (round-robin)")

            # Method 2: Find GPU with enough free memory (5GB)
            best_gpu = self.monitor.find_available_gpu(threshold=1.0, min_free_gb=5.0)  # threshold=1.0 means ignore usage rate
            if best_gpu is not None and best_gpu != gpu_id:
                print(f"[CLIPModelPool] Found GPU {best_gpu} with >=5GB free, switching from {gpu_id}")
                gpu_id = best_gpu
            usage_ratio, free_bytes, total_bytes = self.monitor.get_gpu_memory_info(gpu_id)
            free_gb = free_bytes / (1024**3)

            print(f"[CLIPModelPool] GPU {gpu_id} status: {usage_ratio:.1%} used, {free_gb:.2f} GB free")

            # Try to load model on the selected GPU
            # Note: If we got here, the GPU either has 5GB free (from find_available_gpu)
            # or is the round-robin assigned GPU. CLIP only needs ~500MB.
            try:
                self.device = f"cuda:{gpu_id}"
                self.model = CLIPSimilarityCalculator(self.model_path, self.device)
                print(f"[CLIPModelPool] ✓ Loaded model on GPU {gpu_id} (usage={usage_ratio:.1%}, free={free_gb:.2f}GB)")
                return
            except Exception as e:
                print(f"[CLIPModelPool] ✗ Failed to load model on GPU {gpu_id}: {e}")
                import traceback
                traceback.print_exc()

        # Fallback to CPU
        print("[CLIPModelPool] ⚠⚠⚠ WARNING: Using CPU for CLIP inference (will be VERY SLOW!)")
        print("[CLIPModelPool] ⚠⚠⚠ Expected slowdown: 50-100x compared to GPU")
        self.device = "cpu"
        self.model = CLIPSimilarityCalculator(self.model_path, self.device)

    def get_model(self) -> Tuple[CLIPSimilarityCalculator, str, threading.Lock]:
        """
        Get the CLIP model (always returns the same model)

        Returns:
            Tuple of (model, device, lock)
        """
        if self.model is None:
            # Lazy initialization if not preloaded
            with self.lock:
                if self.model is None:
                    self._init_model()

        return self.model, self.device, self.lock

    def get_status(self) -> Dict:
        """Get status of the model"""
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


# Global instance
_clip_manager = None


def get_clip_manager(model_path: Optional[str] = None, preload: bool = True) -> CLIPModelPool:
    """
    Get or create the global CLIP manager instance

    Args:
        model_path: Path to CLIP model checkpoint
        preload: Whether to preload models on initialization

    Returns:
        CLIPModelPool instance
    """
    global _clip_manager
    if _clip_manager is None:
        _clip_manager = CLIPModelPool(model_path=model_path, preload=preload)
    return _clip_manager


def compute_clip_similarity_batch(
    frames: List[np.ndarray],
    prompt: str,
    batch_size: int = 32
) -> Tuple[List[float], List[int]]:
    """
    Compute CLIP similarity scores for a batch of frames against a text prompt
    Optimized version with TRUE batching (GPU parallelism) and no file IO

    Args:
        frames: List of frame arrays (numpy arrays in RGB format)
        prompt: Text prompt to compare against
        batch_size: Number of frames to process at once (default: 32, increased from 8)

    Returns:
        Tuple of (similarities, indices) where indices are sorted by similarity (descending)
    """
    from PIL import Image

    # Get CLIP manager instance
    clip_manager = get_clip_manager()

    # Get the CLIP model
    model, device, lock = clip_manager.get_model()

    try:
        # Encode text once
        text_features = model.encode_text(prompt)

        similarities = []

        # Process frames in TRUE batches (GPU parallelism)
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            print(f"[CLIPManager] Processing frames {batch_start}-{batch_end}/{len(frames)} (TRUE batch)")

            # Convert numpy arrays to PIL Images
            pil_images = []
            for frame in batch_frames:
                # Convert frame to PIL Image
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                pil_images.append(Image.fromarray(frame))

            # ✨ TRUE BATCH ENCODING: Process all images in one GPU call
            batch_image_features = model.encode_images_batch(pil_images)

            # Compute similarities for the batch
            for image_features in batch_image_features:
                similarity = float(np.dot(image_features, text_features))
                similarities.append(similarity)

        # Sort by similarity (descending)
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

        return similarities, sorted_indices

    except Exception as e:
        print(f"[CLIPManager] Error in compute_clip_similarity_batch: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return uniform similarities
        return [0.5] * len(frames), list(range(len(frames)))