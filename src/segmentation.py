from __future__ import annotations

import time
import numpy as np

from src.base import CVResult

# keyed by name — a single global slot would pin the first checkpoint loaded
_MODELS: dict[str, tuple] = {}


def _lazy_load(name: str):
    if name not in _MODELS:
        from transformers import SamModel, SamProcessor
        _MODELS[name] = (SamModel.from_pretrained(name).eval(),
                         SamProcessor.from_pretrained(name))
    return _MODELS[name]


def run(image: np.ndarray, config: dict) -> CVResult:
    """Segment one region. Prompt with either a point or a box.

    config["point"] = [x, y]           -> segment the object under that point
    config["box"]   = [x1,y1,x2,y2]    -> segment inside that box (e.g. a YOLO box)
    Defaults to the image centre if neither is given.
    """
    import torch
    name = config.get("model_name", "facebook/sam-vit-base")
    model, processor = _lazy_load(name)

    # after the load — the first call downloads ~375MB and that isn't inference
    t0 = time.perf_counter()

    h, w = image.shape[:2]
    box = config.get("box")
    point = config.get("point")

    # HF processors expect RGB, so our image goes in as-is — the opposite of
    # ultralytics in detection.py, which wants BGR
    if box is not None:
        inputs = processor(image, input_boxes=[[box]], return_tensors="pt")
    else:
        if point is None:
            point = [w // 2, h // 2]
        inputs = processor(image, input_points=[[point]], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    # SAM returns 3 candidates per prompt — "the thing under this point" is
    # ambiguous (object / object+support / whole scene) and they are NOT sorted,
    # so pick the one SAM scored highest instead of taking [0] and hoping.
    scores = outputs.iou_scores[0][0]
    best = int(scores.argmax())
    mask = masks[0][0][best].numpy()          # (H, W) bool

    # green overlay
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :, 1] = 255
    alpha = 0.5
    overlay[mask] = (alpha * green[mask] + (1 - alpha) * overlay[mask]).astype(np.uint8)

    return CVResult(
        overlay=overlay,
        data={
            "mask_area_px": int(mask.sum()),
            "mask_fraction": round(float(mask.mean()), 4),
            "mask_score": round(float(scores[best]), 3),
            "prompt": {"box": box} if box is not None else {"point": point},
        },
        meta={
            "module": "segmentation",
            "model": name,
            "runtime_s": round(time.perf_counter() - t0, 3),
        },
    )
