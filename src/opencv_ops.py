"""Phase 1 — classical CV operations behind the CVResult contract.

Every operation is a small function taking (RGB image, config) and returning
(overlay, data). `run` dispatches to the one named in config["operation"].
The dispatch table's keys are also the Streamlit dropdown's options.
"""

import time

import cv2
import numpy as np

from src.base import CVResult
from src.utils import draw_boxes, to_gray, to_rgb


def _odd(k: int) -> int:
    """Kernel sizes must be odd so there's an unambiguous centre pixel."""
    k = int(k)
    return k if k % 2 == 1 else k + 1


MORPH_OPS = {
    "none": None,
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
}


def _binarize(image, config):
    """Grey -> two-tone. Shared by every operation that needs a binary image.

    invert=True makes *dark* pixels the foreground. That is the setting a
    scanned page needs: with the default polarity the white paper becomes the
    object and the ink becomes holes in it.
    """
    gray = to_gray(image)
    flag = cv2.THRESH_BINARY_INV if config.get("invert") else cv2.THRESH_BINARY
    if config.get("threshold", "otsu") == "fixed":
        used, binary = cv2.threshold(gray, int(config.get("thresh", 127)), 255, flag)
    else:
        # The 0 is ignored — Otsu computes its own value from the histogram.
        used, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return binary, float(used)


def _morph(binary, config):
    """Apply the configured morphological operation. 'none' passes through."""
    name = config.get("morph_op", "none")
    if name not in MORPH_OPS:
        raise ValueError(f"Unknown morph_op {name!r}. Available: {sorted(MORPH_OPS)}")

    info = {"morph_op": name}
    if MORPH_OPS[name] is None:
        return binary, info

    kw = max(1, int(config.get("kernel_w", 3)))
    kh = max(1, int(config.get("kernel_h", 3)))
    iterations = max(1, int(config.get("iterations", 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    info.update(kernel=[kw, kh], iterations=iterations)
    return cv2.morphologyEx(binary, MORPH_OPS[name], kernel,
                            iterations=iterations), info


def _min_area(image, config):
    """Contour area floor. The default scales with the image.

    A fixed pixel count means something different on a 4000px photo than on a
    600px scan — on the latter a single character is only a few dozen pixels
    and a generous default silently deletes the entire page of text.
    """
    h, w = image.shape[:2]
    return float(config.get("min_area", max(10.0, 5e-5 * h * w)))

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
    binary, used = _binarize(image, {**config, "threshold": "fixed"})
    return binary, {"threshold_used": used,
                    "inverted": bool(config.get("invert", False))}


def op_threshold_otsu(image, config):
    binary, used = _binarize(image, {**config, "threshold": "otsu"})
    return binary, {"threshold_used": used,
                    "inverted": bool(config.get("invert", False))}


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


def op_morphology(image, config):
    """Binarise, then grow or shrink the white regions with a rectangular kernel.

    erode shrinks, dilate grows; open (erode then dilate) drops specks, close
    (dilate then erode) fills gaps. The kernel's shape is the whole point — a
    wide flat one such as (25, 3) spreads sideways only, so on inverted text it
    smears characters into each other until a line of text is a single blob,
    while leaving the gap between lines intact.
    """
    binary, used = _binarize(image, config)
    out, info = _morph(binary, config)
    return out, {"threshold_used": used,
                 "inverted": bool(config.get("invert", False)),
                 "white_fraction": round(float((out == 255).mean()), 4),
                 **info}


def op_contours(image, config):
    min_area = _min_area(image, config)
    binary, _ = _binarize(image, config)
    binary, morph_info = _morph(binary, config)
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
                     "largest_areas": [round(a, 1) for a in areas[:5]],
                     **morph_info}


def op_text_regions(image, config):
    """The document workflow end to end: binarise, glue characters into lines,
    box what survives.

    Every step is one of the operations above; the only thing this adds is the
    default settings a scanned page wants (dark ink as foreground, a wide flat
    dilation kernel) and boxes instead of contour outlines, because the useful
    output for a document is a list of regions, not their exact silhouettes.

    RETR_LIST rather than the RETR_EXTERNAL that op_contours uses: documents
    nest. A ruled table is one closed outer contour, and under RETR_EXTERNAL
    every line of text inside its cells is discarded as interior detail — which
    on a form or an invoice is most of the page.
    """
    cfg = {"invert": True, "morph_op": "dilate", "kernel_w": 25, "kernel_h": 3,
           **config}
    min_area = _min_area(image, cfg)

    binary, used = _binarize(image, cfg)
    morphed, morph_info = _morph(binary, cfg)
    contours, _ = cv2.findContours(morphed, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        if cv2.contourArea(c) <= min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])
    boxes.sort(key=lambda b: (b[1], b[0]))      # reading order: top-down, left-right

    return draw_boxes(image, boxes), {
        "regions": len(boxes),
        "found": len(contours),
        "min_area": min_area,
        "threshold_used": used,
        "inverted": bool(cfg.get("invert", False)),
        "boxes": boxes,
        **morph_info,
    }


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
    "morphology": op_morphology,
    "contours": op_contours,
    "text_regions": op_text_regions,
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
