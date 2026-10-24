"""Train a YOLO object detector on the pork rasher packaging dataset.

Run from the repository root, on a machine with a GPU (Google Colab or
Kaggle)::

    python meat_quality/train.py                      # defaults: yolo11n.pt at 736
    python meat_quality/train.py --imgsz 640          # another image size
    python meat_quality/train.py --model yolo11s.pt   # another model size
    python meat_quality/train.py --scale 0.2          # gentler scale augmentation

The selected model was trained with ``--scale 0.2``.

Results (trained weights, training curves, validation metrics) are written to
``runs/<model>_<imgsz>/``, e.g. ``runs/yolo11n_736/``, with ``_scale<value>``
added when the scale augmentation is not the default, e.g.
``runs/yolo11n_736_scale0.2/``.

This file only needs Ultralytics, so it can run on Colab or Kaggle without
installing the rest of the package.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "pork-rasher-error-packaging-4" / "data.yaml"

MODEL = "yolo11n.pt"
"""Default starting weights: YOLO11 nano, pretrained on COCO (downloaded automatically)."""

IMAGE_SIZE = 736
"""Default image size: the smallest multiple of 32 that is >= 720, so images are never shrunk."""

EPOCHS = 100
"""Upper limit on epochs; early stopping usually ends training sooner."""

PATIENCE = 20
"""Early stopping: stop if the validation score has not improved for this many epochs."""

SCALE = 0.5
"""Ultralytics' default scale augmentation: images randomly resized by 0.5-1.5x."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the training settings from the command line."""
    parser = argparse.ArgumentParser(description="Train YOLO on the pork rasher dataset.")
    parser.add_argument("--model", default=MODEL,
                        help=f"pretrained weights to start from, e.g. yolo11s.pt (default {MODEL})")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE,
                        help=f"training image size in pixels, a multiple of 32 (default {IMAGE_SIZE})")
    # Scale augmentation randomly resizes each training image by a factor
    # between 1 - scale and 1 + scale. Shrinking makes the already tiny
    # packaging-error boxes even smaller, so a lower value may help them.
    parser.add_argument("--scale", type=float, default=SCALE,
                        help=f"scale augmentation gain, 0 to 1 (default {SCALE})")
    return parser.parse_args(argv)


def run_name(model: str, imgsz: int, scale: float) -> str:
    """Name of the output folder for a run, so runs don't overwrite each other.

    Examples: ``yolo11s_736``, or ``yolo11n_736_scale0.2`` when the scale
    augmentation differs from the default.
    """
    name = f"{Path(model).stem}_{imgsz}"
    if scale != SCALE:
        name += f"_scale{scale}"
    return name


def main(argv: list[str] | None = None) -> None:
    """Train a model with the given settings and save the results under runs/."""
    args = parse_args(argv)
    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        epochs=EPOCHS,
        patience=PATIENCE,
        scale=args.scale,
        project=str(REPO_ROOT / "runs"),
        name=run_name(args.model, args.imgsz, args.scale),
    )


# The __main__ guard is needed because Ultralytics loads data in parallel
# worker processes, which re-import this file on Windows.
if __name__ == "__main__":
    main()
