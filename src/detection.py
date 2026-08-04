
from __future__ import annotations

import time

import cv2
import numpy as np

from src.base import CVResult
from src.utils import draw_boxes


_MODELS: dict[str, object] = {}


def _lazy_load(name: str):
    if name not in _MODELS:
        from ultralytics import YOLO
        _MODELS[name] = YOLO(name)   # 'n' = nano, smallest/fastest
    return _MODELS[name]


def run(image: np.ndarray, config: dict) -> CVResult:
    """Detect objects. image: RGB uint8 (H,W,3).

    conf: confidence threshold raise it for fewer, surer boxes (precision up,
          recall down). iou: NMS overlap threshold ,how aggressively duplicate
          boxes for one object are merged.
    """
    name = config.get("model_name", "yolov8n.pt")
    conf = config.get("conf", 0.25)
    iou = config.get("iou", 0.45)
    model = _lazy_load(name)

    
    t0 = time.perf_counter()

  
    results = model.predict(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    boxes, labels, scores = [], [], []
    for b in results.boxes or []:   
        x1, y1, x2, y2 = b.xyxy[0].tolist()  
        cls_id = int(b.cls[0])
        boxes.append([x1, y1, x2, y2])
        labels.append(results.names[cls_id])
        scores.append(float(b.conf[0]))

    overlay = draw_boxes(image, boxes, labels, scores)   # RGB in, RGB out

    return CVResult(
        overlay=overlay,
        data={
            "objects": [
                {"box": bx, "label": lb, "score": round(sc, 3)}
                for bx, lb, sc in zip(boxes, labels, scores)
            ],
            "count": len(boxes),
        },
        meta={
            "module": "detection",
            "model": name,
            "conf": conf,
            "iou": iou,
            "runtime_s": round(time.perf_counter() - t0, 3),
        },
    )
