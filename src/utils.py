import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Read config.yaml once per process.

    A missing file or a missing key is not an error — every module carries its
    own defaults, so config.yaml only ever overrides them.
    """
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_image(path: str | os.PathLike) -> np.ndarray:
    """Load an image from disk as RGB uint8."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(image: np.ndarray) -> np.ndarray:
    """RGB -> single-channel grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Single-channel -> 3-channel. Streamlit wants 3 channels for every overlay."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def resize_max(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Shrink so the longest side is max_side. Never upscales."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    new_size = (int(w * scale), int(h * scale))   # cv2 wants (W, H)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def draw_boxes(image, boxes, labels=None, scores=None) -> np.ndarray:
    """Draw xyxy boxes with optional labels and scores."""
    out = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        if labels is not None:
            text = labels[i]
            if scores is not None:
                text += f" {scores[i]:.2f}"
            cv2.putText(out, text, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out