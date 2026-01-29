
import os

CLIP_BACKEND = os.environ.get("CLIP_BACKEND", "clip").lower()

if CLIP_BACKEND == "siglip":
    try:
        from .siglip_similarity import (
            compute_clip_similarity,
            get_clip_model,
            CLIPSimilarityCalculator,
            SigLIPSimilarityCalculator
        )
    except ImportError:
        from .clip_similarity import (
            compute_clip_similarity,
            get_clip_model,
            CLIPSimilarityCalculator
        )
else:
    from .clip_similarity import (
        compute_clip_similarity,
        get_clip_model,
        CLIPSimilarityCalculator
    )

__all__ = ['compute_clip_similarity', 'get_clip_model', 'CLIPSimilarityCalculator']
