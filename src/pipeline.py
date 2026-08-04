from __future__ import annotations

import time

from src import classification, clip_module, detection, segmentation
from src.base import CVResult


def run(image, config: dict) -> CVResult:
    t0 = time.perf_counter()
    report = {"objects": [], "scene": {}, "segmentation": {}}
    stages = {}          # per-stage runtimes — the total alone hides which one is slow

    # 1. Detection — what and where
    det = detection.run(image, config.get("detection", {}))
    report["objects"] = det.data["objects"]
    stages["detection"] = det.meta["runtime_s"]

    # 2. Classification — a whole-image scene label (best-effort; skip if no weights)
    try:
        cls = classification.run(image, config.get("classification", {}))
        report["scene"]["classifier_top1"] = cls.data["top1"]
        stages["classification"] = cls.meta["runtime_s"]
    except Exception as e:
        report["scene"]["classifier_top1"] = None
        report["scene"]["classifier_error"] = str(e)

    # 3. Segmentation — mask the highest-confidence detected object, if any
    if det.data["objects"]:
        best = max(det.data["objects"], key=lambda o: o["score"])
        seg_cfg = dict(config.get("segmentation", {}))
        seg_cfg["box"] = best["box"]
        seg = segmentation.run(image, seg_cfg)
        report["segmentation"] = {
            "prompted_by": best["label"],
            "mask_area_px": seg.data["mask_area_px"],
            "mask_fraction": seg.data["mask_fraction"],
            "mask_score": seg.data["mask_score"],
        }
        stages["segmentation"] = seg.meta["runtime_s"]
        overlay = seg.overlay  # richest single overlay to show in the UI
    else:
        overlay = det.overlay

    # 4. CLIP — scene-level tags
    clip = clip_module.run(image, config.get("clip", {}))
    report["scene"]["clip_tags"] = clip.data["scores"][:3]
    stages["clip"] = clip.meta["runtime_s"]

    meta = {
        "module": "pipeline",
        "stages": stages,
        "runtime_s": round(time.perf_counter() - t0, 3),
    }
    report["meta"] = meta

    return CVResult(overlay=overlay, data=report, meta=meta)
