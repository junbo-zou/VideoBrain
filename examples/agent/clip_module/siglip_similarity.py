
import torch
from PIL import Image
import numpy as np
from typing import List, Tuple
import os
import threading
import gc

if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

try:
    import open_clip
    from transformers import AutoTokenizer
except ImportError as e:
    import sys
    sys.exit(1)


_SIGLIP_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()

SIGLIP_MODEL_NAME = "hf-hub:timm/ViT-B-16-SigLIP2"


class SigLIPTokenizerWrapper:

    def __init__(self, hf_tokenizer, max_length: int = 64):
        self.tokenizer = hf_tokenizer
        self.max_length = max_length

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return encoded['input_ids']


def get_siglip_model(device: str = None):
    if device is not None:
        device_obj = torch.device(device)
        device_str = str(device_obj)
    else:
        device_obj = torch.device("cpu")
        device_str = str(device_obj)

    if device_str in _SIGLIP_MODEL_CACHE:
        return _SIGLIP_MODEL_CACHE[device_str] + (device_obj,)

    with _CACHE_LOCK:
        if device_str in _SIGLIP_MODEL_CACHE:
            return _SIGLIP_MODEL_CACHE[device_str] + (device_obj,)


        local_model_dir = os.path.join(os.path.dirname(__file__), "SigLIP2_ViT")

        try:
            local_safetensor = os.path.join(local_model_dir, "open_clip_model.safetensors")
            local_bin = os.path.join(local_model_dir, "open_clip_pytorch_model.bin")
            local_tokenizer_json = os.path.join(local_model_dir, "tokenizer.json")

            if os.path.exists(local_safetensor):
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-16-SigLIP2',
                    pretrained=local_safetensor,
                    device=device_obj
                )
            elif os.path.exists(local_bin):
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-16-SigLIP2',
                    pretrained=local_bin,
                    device=device_obj
                )
            else:
                model, preprocess = open_clip.create_model_from_pretrained(
                    SIGLIP_MODEL_NAME,
                    device=device_obj
                )

            if os.path.exists(local_tokenizer_json):
                tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
                tokenizer = SigLIPTokenizerWrapper(tokenizer)
            else:
                tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP2')

        except Exception as e:
            model, preprocess = open_clip.create_model_from_pretrained(
                SIGLIP_MODEL_NAME,
                device=device_obj
            )
            tokenizer = open_clip.get_tokenizer(SIGLIP_MODEL_NAME)

        model.eval()

        inference_lock = threading.Lock()

        _SIGLIP_MODEL_CACHE[device_str] = (model, preprocess, tokenizer, inference_lock)

        return model, preprocess, tokenizer, inference_lock, device_obj


get_clip_model = get_siglip_model


class SigLIPSimilarityCalculator:

    def __init__(self, model_path: str = None, device: str = None):
        self.model, self.preprocess, self.tokenizer, self.inference_lock, self.device = get_siglip_model(device)

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
        return self.encode_image_pil(image)

    def encode_text(self, text: str) -> np.ndarray:
        text_tokens = self.tokenizer([text]).to(self.device)

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


CLIPSimilarityCalculator = SigLIPSimilarityCalculator


def compute_clip_similarity(
    frame_paths: List[str],
    question: str,
    model_path: str = None,
    device: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    calculator = SigLIPSimilarityCalculator(model_path, device)
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

        calculator = SigLIPSimilarityCalculator()
        similarity = calculator.compute_similarity(image_path, text)
