from __future__ import annotations

import time
import numpy as np
import torch
import timm
from PIL import Image

from src.base import CVResult

# keyed by weights path, so Streamlit reruns don't reload 111MB every interaction
_MODELS: dict[str, tuple] = {}


def _lazy_load(weights_path: str):
    if weights_path not in _MODELS:
        # the checkpoint carries its own architecture and class order — a list
        # retyped here would go stale the moment the classifier is retrained
        ckpt = torch.load(weights_path, map_location="cpu")
        classes = ckpt["classes"]
        model = timm.create_model(ckpt["model"], pretrained=False,
                                  num_classes=len(classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        cfg = timm.data.resolve_data_config({}, model=model)
        _MODELS[weights_path] = (model, timm.data.create_transform(**cfg),
                                 classes, ckpt["model"])
    return _MODELS[weights_path]


def run(image: np.ndarray, config: dict) -> CVResult:
    """Classify a whole image. image: RGB uint8 (H,W,3)."""
    weights_path = config.get("weights_path", "models/classifier.pt")
    model, transform, classes, model_name = _lazy_load(weights_path)

    t0 = time.perf_counter()
    tensor = transform(Image.fromarray(image)).unsqueeze(0)  # (1,C,H,W) — batch dim
    with torch.no_grad():                                    # no gradients at inference
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]              # logits -> probabilities

    topk = torch.topk(probs, k=min(5, len(classes)))
    predictions = [
        {"label": classes[i], "score": round(float(probs[i]), 4)}
        for i in topk.indices.tolist()
    ]

    return CVResult(
        overlay=None,  # classification has no natural overlay; the UI shows the table
        data={"predictions": predictions, "top1": predictions[0]["label"]},
        meta={
            "module": "classification",
            "model": model_name,
            "weights": weights_path,
            "classes": classes,   # the UI names the domain from this, not a copy

            "runtime_s": round(time.perf_counter() - t0, 3),
        },
    )
