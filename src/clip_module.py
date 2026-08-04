

from __future__ import annotations

import time
import numpy as np

from src.base import CVResult
from src.utils import load_config

# keyed by name — a single global slot would pin the first checkpoint loaded
_MODELS: dict[str, tuple] = {}

FALLBACK_LABELS = ["a scanned document", "a page of text", "a photograph",
                   "an object"]


def default_labels() -> list[str]:
    """The label set config.yaml nominates, or a generic one if it says nothing."""
    cfg = load_config().get("clip", {})
    sets = cfg.get("label_sets", {})
    return sets.get(cfg.get("default_label_set", "general")) or FALLBACK_LABELS


def _lazy_load(name: str):
    if name not in _MODELS:
        from transformers import CLIPModel, CLIPProcessor
        _MODELS[name] = (CLIPModel.from_pretrained(name).eval(),
                         CLIPProcessor.from_pretrained(name))
    return _MODELS[name]


def run(image: np.ndarray, config: dict) -> CVResult:
    """Zero-shot classify against a runtime label set.

    config["labels"] = ["a forest", "a city", "a river", "farmland"]
    Prompt wording matters: "a photo of a {label}" beats a bare label. The
    module wraps labels in that template unless you set config["raw_labels"].
    Labels and model default to config.yaml.
    """
    import torch
    name = (config.get("model_name")
            or load_config().get("clip", {}).get("model_name",
                                                 "openai/clip-vit-base-patch32"))
    model, processor = _lazy_load(name)

    # after the load — the first call downloads ~600MB and that isn't inference
    t0 = time.perf_counter()

    labels = config.get("labels") or default_labels()
    if not config.get("raw_labels", False):
        prompts = [f"a photo of {l}" for l in labels]
    else:
        prompts = labels

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image is already image-vs-each-text similarity; softmax -> a
        # distribution over your labels.
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    ranked = sorted(
        [{"label": labels[i], "score": float(probs[i])} for i in range(len(labels))],
        key=lambda d: d["score"],
        reverse=True,
    )

    return CVResult(
        overlay=None,
        data={"scores": ranked, "top": ranked[0]["label"]},
        meta={
            "module": "clip",
            "model": name,
            "labels": len(labels),
            "runtime_s": round(time.perf_counter() - t0, 3),
        },
    )
