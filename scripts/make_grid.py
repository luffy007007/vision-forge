"""Run every opencv_ops operation on one sample image and save a labelled grid.

    python scripts\\make_grid.py  
    python scripts\\make_grid.py data/samples/text.jpg

Output: assets/opencv_grid.png — this goes straight into the README.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # no GUI window, save to disk
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))       # import src,works when run directly

from src.opencv_ops import run
from src.utils import load_image, resize_max

DEFAULT_IMAGE = ROOT / "data" / "samples" / "urbancity.jpg"
OUTPUT = ROOT / "assets" / "opencv_grid.png"

# One config per panel. Settings chosen to make the differences visible rather
# than to be optimal the grid is a teaching image.
PANELS = [
    {"operation": "grayscale"},
    {"operation": "blur", "ksize": 15},
    {"operation": "box_blur", "ksize": 5},
    {"operation": "sharpen", "strength": 5},
    {"operation": "sobel", "direction": "vertical"},
    {"operation": "threshold_fixed", "thresh": 127},
    {"operation": "threshold_otsu"},
    {"operation": "threshold_adaptive", "block_size": 11, "C": 2},
    {"operation": "canny", "low": 50, "high": 150},
    {"operation": "contours", "min_area": 500},
]


def caption(config: dict, result) -> str:
    """Operation name plus the one number worth reading off the grid."""
    name = config["operation"]
    d = result.data
    if name == "threshold_otsu":
        return f"threshold_otsu (chose {d['threshold_used']:.0f})"
    if name == "canny":
        return f"canny ({d['low']}, {d['high']})"
    if name == "contours":
        return f"contours ({d['kept']} of {d['found']} kept)"
    if name in ("blur", "box_blur"):
        return f"{name} (k={d['ksize']})"
    if name == "threshold_adaptive":
        return f"threshold_adaptive (block={d['block_size']})"
    return name


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    image = resize_max(load_image(path), 800)

    tiles = [(f"original — {path.name}", image)]
    for config in PANELS:
        result = run(image, config)
        print(f"{config['operation']:<20} {result.meta['runtime_s']:>7.4f}s  {result.data}")
        tiles.append((caption(config, result), result.overlay))

    ncols = 4
    nrows = -(-len(tiles) // ncols)          # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))

    for ax, (title, img) in zip(axes.flat, tiles):
        ax.imshow(img)                       # already RGB, 3-channel
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes.flat[len(tiles):]:        # hide unused cells
        ax.axis("off")

    fig.suptitle("Phase 1 — classical CV operations", fontsize=14)
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()
