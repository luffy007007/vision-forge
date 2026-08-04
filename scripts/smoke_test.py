

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))       # import src, works when run directly

from src import opencv_ops
from src.base import CVResult
from src.utils import load_config, load_image

PHOTO = ROOT / "data" / "samples" / "urbancity.jpg"
DOCUMENT = ROOT / "data" / "samples" / "text.jpg"

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def check_contract(name: str, result, image) -> None:
    """Every module returns the same thing — that is the one structural rule."""
    ok = isinstance(result, CVResult)
    check(f"{name}: returns CVResult", ok)
    if not ok:
        return

    if result.overlay is not None:
        o = result.overlay
        check(f"{name}: overlay is RGB uint8 (H,W,3)",
              o.ndim == 3 and o.shape[2] == 3 and o.dtype == np.uint8,
              f"got {o.shape} {o.dtype}")
        check(f"{name}: overlay matches input size",
              o.shape[:2] == image.shape[:2], f"{o.shape[:2]} vs {image.shape[:2]}")
    try:
        json.dumps(result.data)
        json.dumps(result.meta)
        check(f"{name}: data and meta are JSON-serialisable", True)
    except TypeError as e:
        check(f"{name}: data and meta are JSON-serialisable", False, str(e))


def test_operations(image) -> None:
    """Every registered operation runs, and none of them touch the input."""
    print("\nopencv_ops — every operation in the registry")
    before = image.copy()
    for name in opencv_ops.OPERATIONS:
        try:
            result = opencv_ops.run(image, {"operation": name})
        except Exception as e:
            check(f"{name}: runs", False, f"{type(e).__name__}: {e}")
            continue
        check_contract(name, result, image)
    check("input image not mutated", np.array_equal(image, before))

    try:
        opencv_ops.run(image, {"operation": "nope"})
        check("unknown operation raises", False)
    except ValueError:
        check("unknown operation raises", True)


def test_document(document) -> None:
    """The document workflow: polarity, morphology, and the area floor.

    These are the three things that were wrong before Document mode existed,
    so they get assertions rather than a smoke check.
    """
    print("\ndocument workflow — data/samples/text.jpg")
    h, w = document.shape[:2]
    page_area = h * w

    default = opencv_ops.run(document, {"operation": "contours"})
    inverted = opencv_ops.run(document, {"operation": "contours", "invert": True})
    check("inverting finds the ink, not the page",
          inverted.data["found"] > default.data["found"] * 5,
          f"{inverted.data['found']} vs {default.data['found']}")

    check("default min_area scales with the image",
          10 <= default.data["min_area"] <= 0.001 * page_area,
          f"got {default.data['min_area']}")

    lines = opencv_ops.run(document, {
        "operation": "text_regions", "threshold": "otsu", "invert": True,
        "morph_op": "dilate", "kernel_w": 25, "kernel_h": 3, "min_area": 15,
    })
    check_contract("text_regions", lines, document)
    check("text lines found", lines.data["regions"] >= 5,
          f"got {lines.data['regions']}")

    # A ruled border or table frame is one contour that legitimately spans the
    # page, so the property worth asserting is that the page was broken into
    # line-sized pieces — not that nothing large came back.
    areas = sorted((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in lines.data["boxes"])
    median = areas[len(areas) // 2] if areas else 0
    check("regions are text-line sized", 0 < median < 0.05 * page_area,
          f"median {median} of {page_area}")
    check("page-spanning regions are the exception",
          sum(a > 0.5 * page_area for a in areas) <= 2,
          f"{sum(a > 0.5 * page_area for a in areas)} of {len(areas)}")

    check("dilation merges characters into fewer, bigger blobs",
          lines.data["regions"] < inverted.data["found"],
          f"{lines.data['regions']} vs {inverted.data['found']}")

    none_morph = opencv_ops.run(document, {"operation": "morphology",
                                           "morph_op": "none"})
    otsu = opencv_ops.run(document, {"operation": "threshold_otsu"})
    check("morphology 'none' is a plain threshold",
          np.array_equal(none_morph.overlay, otsu.overlay))


def test_config() -> None:
    print("\nconfig.yaml")
    from src import clip_module
    config = load_config()
    check("config.yaml loads", isinstance(config, dict) and bool(config))
    labels = clip_module.default_labels()
    check("CLIP defaults cover documents",
          any("document" in l or "text" in l for l in labels), str(labels))


def test_models(image) -> None:
    """The deep modules. Each is independent — one missing weight file should
    not stop the rest from being checked."""
    print("\nmodels — first run downloads weights")
    from src import classification, clip_module, detection, pipeline, segmentation

    jobs = [
        ("detection", detection, {}),
        ("classification", classification, {}),
        ("segmentation", segmentation, {"point": [image.shape[1] // 2,
                                                  image.shape[0] // 2]}),
        ("clip", clip_module, {}),
        ("pipeline", pipeline, {}),
    ]
    for name, module, config in jobs:
        try:
            check_contract(name, module.run(image, config), image)
        except Exception as e:
            check(f"{name}: runs", False, f"{type(e).__name__}: {e}")


def main() -> None:
    fast = "--fast" in sys.argv
    image = load_image(PHOTO)
    document = load_image(DOCUMENT)

    test_operations(image)
    test_document(document)
    test_config()
    if fast:
        print("\nmodels — SKIPPED (--fast)")
    else:
        test_models(image)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
