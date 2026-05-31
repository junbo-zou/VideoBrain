import torch
from PIL import Image
import numpy as np
from typing import List, Tuple
import os
import threading
import gc

# Import CLIP with error handling
try:
    import clip
except ImportError as e:
    import sys
    print("=" * 80)
    print("ERROR: CLIP library not found!")
    print("Please install it with:")
    print("  pip uninstall clip -y")
    print("  pip install ftfy regex tqdm")
    print("  pip install git+https://github.com/openai/CLIP.git")
    print("=" * 80)
    sys.exit(1)


# Global model cache: {device_str: (model, preprocess, inference_lock)}
_CLIP_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_clip_model(device: str = None):
    """Get or create CLIP model for the specified device (thread-safe, singleton per GPU).

    Args:
        device: Device string (e.g., 'cuda:0', 'cuda:1', 'cpu'). If None, defaults to CPU for safety.

    Returns:
        Tuple of (model, preprocess, inference_lock, device)
    """
    # Determine device
    if device is not None:
        device_obj = torch.device(device)
        device_str = str(device_obj)
    else:
        # Try to use GPU if available, as CPU is extremely slow
        if torch.cuda.is_available():
            device_obj = torch.device("cuda:0")
            device_str = str(device_obj)
            print(f"[CLIP Model Pool] No device specified, auto-selecting GPU: {device_str}")
        else:
            device_obj = torch.device("cpu")
            device_str = str(device_obj)
            print(f"[CLIP Model Pool] WARNING: No GPU available, using CPU (will be SLOW!)")

    # Check cache (double-checked locking for performance)
    if device_str in _CLIP_MODEL_CACHE:
        return _CLIP_MODEL_CACHE[device_str] + (device_obj,)

    # Acquire lock to create model
    with _CACHE_LOCK:
        # Double-check after acquiring lock
        if device_str in _CLIP_MODEL_CACHE:
            return _CLIP_MODEL_CACHE[device_str] + (device_obj,)

        print(f"[CLIP Model Pool] Loading CLIP ViT-B/16 on {device_str}...")

        # Try to load from local CLIP_ViT directory first
        local_clip_path = os.path.join(os.path.dirname(__file__), "CLIP_ViT", "ViT-B-16.pt")

        try:
            if os.path.exists(local_clip_path):
                print(f"[CLIP Model Pool] Loading from local path: {local_clip_path}")
                model, preprocess = clip.load(local_clip_path, device=device_obj, jit=False)
            else:
                print(f"[CLIP Model Pool] Local model not found, downloading ViT-B/16...")
                model, preprocess = clip.load("ViT-B/16", device=device_obj, jit=False)
        except Exception as e:
            print(f"[CLIP Model Pool] Failed with jit=False: {e}")
            # Fallback for older CLIP versions or if local model is incompatible
            if os.path.exists(local_clip_path):
                model, preprocess = clip.load(local_clip_path, device=device_obj)
            else:
                model, preprocess = clip.load("ViT-B/16", device=device_obj)

        model.eval()

        # Create a lock for this model's inference operations
        # This prevents concurrent threads from accessing the same model simultaneously
        inference_lock = threading.Lock()

        _CLIP_MODEL_CACHE[device_str] = (model, preprocess, inference_lock)
        print(f"[CLIP Model Pool] ✓ Model loaded and cached for {device_str}")

        return model, preprocess, inference_lock, device_obj


class CLIPSimilarityCalculator:
    """Calculate image-text similarity using CLIP with global model pooling."""

    def __init__(self, model_path: str = None, device: str = None):
        """Initialize CLIP model (reuses global model if available).

        Args:
            model_path: Path to CLIP model directory (ignored, kept for compatibility)
            device: Device to use (e.g., 'cuda:0', 'cuda:1', etc.). If None, auto-detect.
        """
        # Get shared model from global pool
        self.model, self.preprocess, self.inference_lock, self.device = get_clip_model(device)

    def encode_image_pil(self, image: Image.Image) -> np.ndarray:
        """Encode PIL Image to feature vector (thread-safe, no file IO).

        Args:
            image: PIL Image object in RGB format

        Returns:
            Normalized image feature vector
        """
        # Preprocess image (already PIL Image, no need to load)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # For CPU: PyTorch handles multi-threading internally, no lock needed
        # For GPU: Lock prevents concurrent access to same GPU model
        if self.device.type == 'cuda':
            # GPU mode: use lock to prevent concurrent access
            with self.inference_lock:
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensor)
                    # Convert to numpy immediately within no_grad context
                    image_features_np = image_features.cpu().numpy().squeeze()

                # Explicitly delete CLIP's tensors (ONLY CLIP's, not vLLM's)
                del image_tensor, image_features
        else:
            # CPU mode: no lock needed, PyTorch uses multi-threading
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Convert to numpy immediately within no_grad context
                image_features_np = image_features.cpu().numpy().squeeze()

            # Explicitly delete CLIP's tensors
            del image_tensor, image_features

        # Normalize
        image_features_np = image_features_np / np.linalg.norm(image_features_np)

        return image_features_np

    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image to feature vector (thread-safe).

        Args:
            image_path: Path to image file

        Returns:
            Normalized image feature vector
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # For CPU: PyTorch handles multi-threading internally, no lock needed
        # For GPU: Lock prevents concurrent access to same GPU model
        if self.device.type == 'cuda':
            # GPU mode: use lock to prevent concurrent access
            with self.inference_lock:
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensor)
                    # Convert to numpy immediately within no_grad context
                    image_features_np = image_features.cpu().numpy().squeeze()

                # Explicitly delete CLIP's tensors (ONLY CLIP's, not vLLM's)
                del image_tensor, image_features
        else:
            # CPU mode: no lock needed, PyTorch uses multi-threading
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Convert to numpy immediately within no_grad context
                image_features_np = image_features.cpu().numpy().squeeze()

            # Explicitly delete CLIP's tensors
            del image_tensor, image_features

        # Normalize
        image_features_np = image_features_np / np.linalg.norm(image_features_np)

        return image_features_np

    def encode_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL Images to feature vectors (thread-safe, optimized).

        Args:
            images: List of PIL Image objects in RGB format

        Returns:
            Normalized image feature matrix (num_images, feature_dim)
        """
        if not images:
            return np.array([])

        # Preprocess all images and stack into a batch
        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        # For CPU: PyTorch handles multi-threading internally, no lock needed
        # For GPU: Lock prevents concurrent access to same GPU model
        if self.device.type == 'cuda':
            # GPU mode: use lock to prevent concurrent access
            with self.inference_lock:
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensors)
                    # Convert to numpy immediately within no_grad context
                    image_features_np = image_features.cpu().numpy()

                # Explicitly delete CLIP's tensors
                del image_tensors, image_features
        else:
            # CPU mode: no lock needed, PyTorch uses multi-threading
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors)
                # Convert to numpy immediately within no_grad context
                image_features_np = image_features.cpu().numpy()

            # Explicitly delete CLIP's tensors
            del image_tensors, image_features

        # Normalize each feature vector
        norms = np.linalg.norm(image_features_np, axis=1, keepdims=True)
        image_features_np = image_features_np / norms

        return image_features_np

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to feature vector (thread-safe).

        Args:
            text: Text string

        Returns:
            Normalized text feature vector
        """
        # Tokenize text with truncation enabled to handle long prompts
        # CLIP has a hard limit of 77 tokens, truncate=True automatically handles this
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)

        # For CPU: PyTorch handles multi-threading internally, no lock needed
        # For GPU: Lock prevents concurrent access to same GPU model
        if self.device.type == 'cuda':
            # GPU mode: use lock to prevent concurrent access
            with self.inference_lock:
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    # Convert to numpy immediately within no_grad context
                    text_features_np = text_features.cpu().numpy().squeeze()

                # Explicitly delete CLIP's tensors (ONLY CLIP's, not vLLM's)
                del text_tokens, text_features
        else:
            # CPU mode: no lock needed, PyTorch uses multi-threading
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                # Convert to numpy immediately within no_grad context
                text_features_np = text_features.cpu().numpy().squeeze()

            # Explicitly delete CLIP's tensors
            del text_tokens, text_features

        # Normalize
        text_features_np = text_features_np / np.linalg.norm(text_features_np)

        return text_features_np

    def compute_similarity(self, image_path: str, text: str) -> float:
        """Compute cosine similarity between image and text.

        Args:
            image_path: Path to image file
            text: Text string

        Returns:
            Cosine similarity score (after L2 normalization)
        """
        # Get features
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(text)

        # Compute cosine similarity (dot product of normalized vectors)
        similarity = np.dot(image_features, text_features)

        return float(similarity)


def compute_clip_similarity(
    frame_paths: List[str],
    question: str,
    model_path: str = None,
    device: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CLIP similarity between frames and question.

    Args:
        frame_paths: List of frame image paths
        question: Question text
        model_path: Path to CLIP model directory (ignored, kept for compatibility)
        device: Device to use (e.g., 'cuda:0', 'cuda:1', etc.). If None, auto-detect.

    Returns:
        Tuple of (image_features, similarities)
        - image_features: Image feature matrix (num_frames x feature_dim)
        - similarities: Similarity scores for each frame (num_frames,)
    """
    # Reuse global model
    calculator = CLIPSimilarityCalculator(model_path, device)

    # Encode text once
    text_features = calculator.encode_text(question)

    # Encode all images and compute similarities
    image_features_list = []
    similarities = []

    print(f"Computing CLIP similarity for {len(frame_paths)} frames on {calculator.device}...")

    for i, image_path in enumerate(frame_paths):
        # Encode image
        image_features = calculator.encode_image(image_path)
        image_features_list.append(image_features)

        # Compute similarity
        similarity = np.dot(image_features, text_features)
        similarities.append(similarity)

        # Show progress periodically
        if (i + 1) % 50 == 0 or (i + 1) == len(frame_paths):
            print(f"  Processed {i + 1}/{len(frame_paths)} frames")

    # Stack features
    image_features_matrix = np.stack(image_features_list, axis=0)
    similarities_array = np.array(similarities)

    print(f"\nCLIP similarity statistics:")
    print(f"  Mean similarity: {np.mean(similarities_array):.4f}")
    print(f"  Std similarity: {np.std(similarities_array):.4f}")
    print(f"  Min similarity: {np.min(similarities_array):.4f}")
    print(f"  Max similarity: {np.max(similarities_array):.4f}")

    # Clean up CLIP's memory (IMPORTANT: Only CLIP's, never vLLM's)
    del calculator, image_features_list, text_features

    # Trigger Python garbage collection for CLIP objects
    # This is safe - only collects unreferenced Python objects, doesn't touch vLLM
    gc.collect()

    return image_features_matrix, similarities_array


if __name__ == "__main__":
    # Test CLIP similarity
    import sys

    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        text = sys.argv[2]

        calculator = CLIPSimilarityCalculator()
        similarity = calculator.compute_similarity(image_path, text)

        print(f"Image: {image_path}")
        print(f"Text: {text}")
        print(f"CLIP Similarity: {similarity:.6f}")
    else:
        print("Usage: python clip_similarity.py <image_path> <text>")
