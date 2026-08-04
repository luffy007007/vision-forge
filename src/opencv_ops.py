"""Phase 1 — classical CV operations behind the CVResult contract.

Every operation is a small function taking (RGB image, config) and returning
(overlay, data). `run` dispatches to the one named in config["operation"].
The dispatch table's keys are also the Streamlit dropdown's options.
"""

import time

import cv2
import numpy as np

from src.base import CVResult
from src.utils import to_gray, to_rgb


def _odd(k: int) -> int:
    """Kernel sizes must be odd so there's an unambiguous centre pixel."""
    k = int(k)
    return k if k % 2 == 1 else k + 1

# Each returns (overlay, data). Overlays are left single-channel here; `run`
# normalises them to 3 channels on the way out.


def op_grayscale(image, config):
    gray = to_gray(image)
    return gray, {"mean_intensity": float(gray.mean())}


def op_blur(image, config):
    k = _odd(config.get("ksize", 15))
    # sigma=0 tells OpenCV to derive it from the kernel size
    return cv2.GaussianBlur(image, (k, k), 0), {"ksize": k}


def op_box_blur(image, config):
    """The same operation as op_blur, with a kernel you can read.

    Values sum to 1 (hence the /(k*k)) so total intensity is preserved — a
    kernel summing to more or less would brighten or darken the image rather
    than average it. This is a CNN convolution with hand-picked numbers.
    """
    k = _odd(config.get("ksize", 5))
    kernel = np.ones((k, k), np.float32) / (k * k)
    return cv2.filter2D(image, -1, kernel), {"ksize": k,
                                             "kernel_sum": float(kernel.sum())}


def op_sharpen(image, config):
    strength = float(config.get("strength", 5.0))
    kernel = np.array([[0, -1, 0],
                       [-1, strength, -1],
                       [0, -1, 0]], np.float32)
    return cv2.filter2D(image, -1, kernel), {"centre": strength,
                                             "kernel_sum": float(kernel.sum())}


def op_sobel(image, config):
    """Hand-built edge detector. Transposing the kernel swaps which edges respond."""
    direction = config.get("direction", "vertical")
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], np.float32)
    if direction == "horizontal":
        kernel = kernel.T
    gray = to_gray(image).astype(np.float32)
    edges = cv2.filter2D(gray, -1, kernel)
    # Gradients are signed; take magnitude and clip back into uint8 for display.
    return cv2.convertScaleAbs(edges), {"direction": direction}


def op_threshold_fixed(image, config):
    thresh = int(config.get("thresh", 127))
    used, binary = cv2.threshold(to_gray(image), thresh, 255, cv2.THRESH_BINARY)
    return binary, {"threshold_used": float(used)}


def op_threshold_otsu(image, config):
    # The 0 is ignored — Otsu computes its own value from the histogram.
    used, binary = cv2.threshold(to_gray(image), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, {"threshold_used": float(used)}


def op_threshold_adaptive(image, config):
    block = max(3, _odd(config.get("block_size", 11)))
    C = int(config.get("C", 2))
    binary = cv2.adaptiveThreshold(
        to_gray(image), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block,     # neighbourhood that defines "local"
        C=C,                 # shifts the threshold away from the local mean
    )
    return binary, {"block_size": block, "C": C}


def op_canny(image, config):
    low = int(config.get("low", 50))
    high = int(config.get("high", 150))
    pre_blur = bool(config.get("pre_blur", True))
    gray = to_gray(image)
    if pre_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, low, high)
    return edges, {"low": low, "high": high, "pre_blur": pre_blur,
                   "edge_pixels": int(np.count_nonzero(edges))}


def op_contours(image, config):
    min_area = float(config.get("min_area", 500))
    _, binary = cv2.threshold(to_gray(image), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # RETR_EXTERNAL: outermost contours only (a donut gives the outside, not
    # the hole). CHAIN_APPROX_SIMPLE: straight runs stored as endpoints.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > min_area]

    overlay = image.copy()          # drawContours writes in place
    cv2.drawContours(overlay, big, -1, (0, 255, 0), 2)

    areas = sorted((float(cv2.contourArea(c)) for c in big), reverse=True)
    return overlay, {"found": len(contours),
                     "kept": len(big),
                     "min_area": min_area,
                     "largest_areas": [round(a, 1) for a in areas[:5]]}


OPERATIONS = {
    "grayscale": op_grayscale,
    "blur": op_blur,
    "box_blur": op_box_blur,
    "sharpen": op_sharpen,
    "sobel": op_sobel,
    "threshold_fixed": op_threshold_fixed,
    "threshold_otsu": op_threshold_otsu,
    "threshold_adaptive": op_threshold_adaptive,
    "canny": op_canny,
    "contours": op_contours,
}


def run(image: np.ndarray, config: dict) -> CVResult:
    """Apply one classical CV operation.

    image  : RGB, uint8, (H, W, 3)
    config : {"operation": <one of OPERATIONS>, ...operation-specific settings}
    """
    name = config.get("operation", "canny")
    if name not in OPERATIONS:
        raise ValueError(f"Unknown operation {name!r}. "
                         f"Available: {sorted(OPERATIONS)}")

    start = time.perf_counter()
    overlay, data = OPERATIONS[name](image, config)
    runtime = time.perf_counter() - start

    return CVResult(
        overlay=to_rgb(overlay),    # single-channel results back to 3 channels
        data=data,
        meta={"module": "opencv_ops",
              "operation": name,
              "runtime_s": round(runtime, 4)},
    )
