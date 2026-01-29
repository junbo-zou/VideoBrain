import torch
from PIL import Image
import numpy as np
from typing import List, Tuple
import os
import threading
import gc

try:
    import clip
except ImportError as e:
    import sys
    sys.exit(1)


_CLIP_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_clip_model(device: str = None):
    if device is not None:
        device_obj = torch.device(device)
        device_str = str(device_obj)
    else:
        device_obj = torch.device("cpu")
        device_str = str(device_obj)

    if device_str in _CLIP_MODEL_CACHE:
        return _CLIP_MODEL_CACHE[device_str] + (device_obj,)

    with _CACHE_LOCK:
        if device_str in _CLIP_MODEL_CACHE:
            return _CLIP_MODEL_CACHE[device_str] + (device_obj,)


        local_clip_path = os.path.join(os.path.dirname(__file__), "CLIP_ViT", "ViT-B-16.pt")

        try:
            if os.path.exists(local_clip_path):
                model, preprocess = clip.load(local_clip_path, device=device_obj, jit=False)
            else:
                model, preprocess = clip.load("ViT-B/16", device=device_obj, jit=False)
        except Exception as e:
            if os.path.exists(local_clip_path):
                model, preprocess = clip.load(local_clip_path, device=device_obj)
            else:
                model, preprocess = clip.load("ViT-B/16", device=device_obj)

        model.eval()

        inference_lock = threading.Lock()

        _CLIP_MODEL_CACHE[device_str] = (model, preprocess, inference_lock)

        return model, preprocess, inference_lock, device_obj


class CLIPSimilarityCalculator:

    def __init__(self, model_path: str = None, device: str = None):
        self.model, self.preprocess, self.inference_lock, self.device = get_clip_model(device)

    def encode_image_pil(self, image: Image.Image) -> np.ndarray:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        if self.device.type == 'cuda':
            with self.inference_lock:
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensor)
                    image_features_np = image_features.cpu().numpy().squeeze()

                del image_tensor, image_features
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features_np = image_features.cpu().numpy().squeeze()

            del image_tensor, image_features

        image_features_np = image_features_np / np.linalg.norm(image_features_np)

        return image_features_np

    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        if self.device.type == 'cuda':
            with self.inference_lock:
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensor)
                    image_features_np = image_features.cpu().numpy().squeeze()

                del image_tensor, image_features
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features_np = image_features.cpu().numpy().squeeze()

            del image_tensor, image_features

        image_features_np = image_features_np / np.linalg.norm(image_features_np)

        return image_features_np

    def encode_text(self, text: str) -> np.ndarray:
        text_tokens = clip.tokenize([text]).to(self.device)

        if self.device.type == 'cuda':
            with self.inference_lock:
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features_np = text_features.cpu().numpy().squeeze()

                del text_tokens, text_features
        else:
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features_np = text_features.cpu().numpy().squeeze()

            del text_tokens, text_features

        text_features_np = text_features_np / np.linalg.norm(text_features_np)

        return text_features_np

    def compute_similarity(self, image_path: str, text: str) -> float:
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(text)

        similarity = np.dot(image_features, text_features)

        return float(similarity)


def compute_clip_similarity(
    frame_paths: List[str],
    question: str,
    model_path: str = None,
    device: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    calculator = CLIPSimilarityCalculator(model_path, device)

    text_features = calculator.encode_text(question)

    image_features_list = []
    similarities = []


    for i, image_path in enumerate(frame_paths):
        image_features = calculator.encode_image(image_path)
        image_features_list.append(image_features)

        similarity = np.dot(image_features, text_features)
        similarities.append(similarity)

    image_features_matrix = np.stack(image_features_list, axis=0)
    similarities_array = np.array(similarities)


    del calculator, image_features_list, text_features

    gc.collect()

    return image_features_matrix, similarities_array


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        text = sys.argv[2]

        calculator = CLIPSimilarityCalculator()
        similarity = calculator.compute_similarity(image_path, text)
