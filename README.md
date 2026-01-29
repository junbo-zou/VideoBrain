# Anonymous Code Submission for ICML 2026

This supplementary code package contains the core implementation of our proposed framework for long-video reasoning.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Download Vision Model

Download the required SigLIP model for frame similarity computation:

- Model name: google/siglip-so400m-patch14-384
- Place in: `examples/agent/clip_module/SigLIP2_ViT/`

### Inference

```bash
python examples/agent/infer.py
```

Edit the CONFIG section in the script to set:

- API_URL: Your model serving endpoint
- MODEL_NAME: Path to your trained model
- TARGET_JSON_PATH: Input questions file
- BASE_VIDEO_DIR: Directory containing videos
- OUTPUT_JSON_PATH: Where to save results

Output format:

```json
[
  {
    "video": "video_id",
    "question": "question text",
    "answer": "model answer"
  }
]
```

## 📦 Dependencies

Core requirements:

```
Python >= 3.10
PyTorch >= 2.4.0
vLLM == 0.11.1
transformers == 4.57.1
```

Additional dependencies are listed in `requirements.txt`.

## 🙏 Acknowledgements

This work builds upon open-source frameworks including verl, DeepEyes, PyTorch, Transformers, vLLM, and vision-language models. All third-party code retains original licenses.

## 📝 Anonymization Note

This submission has been anonymized for review. All author information, paper title, institution details, and identifying information have been removed. Full attribution will be provided upon acceptance.

---

**Anonymous Submission for ICML 2026**
