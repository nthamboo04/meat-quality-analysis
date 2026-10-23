"""Train a YOLO object detector on the pork rasher packaging dataset.

Run from the repository root (on Colab with a GPU)::

    python meat_quality/train.py                      # defaults: yolo11n.pt at 736
    python meat_quality/train.py --imgsz 640          # another image size
    python meat_quality/train.py --model yolo11s.pt   # another model size
    python meat_quality/train.py --scale 0.2          # gentler scale augmentation

Results (trained weights, training curves, validation metrics) are written to
runs/<model>_<imgsz>/, e.g. runs/yolo11n_736/, with _scale<value> added when
the scale augmentation is not the default, e.g. runs/yolo11n_736_scale0.2/.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "pork-rasher-error-packaging-4" / "data.yaml"

# Training settings
MODEL = "yolo11n.pt"  # default: nano model, pretrained on COCO (downloaded automatically)
IMAGE_SIZE = 736      # default: smallest multiple of 32 >= 720, so images are never shrunk
EPOCHS = 100          # upper limit; early stopping usually ends training sooner
PATIENCE = 20         # stop if validation mAP has not improved for 20 epochs
SCALE = 0.5           # default (Ultralytics' own): images randomly resized by 0.5-1.5x

# The __main__ guard is needed because Ultralytics loads data in parallel
# worker processes, which re-import this file on Windows.
if __name__ == "__main__":
    # Read the settings from the command line, so different settings can be
    # compared without editing this file.
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
    args = parser.parse_args()

    # Run name, e.g. yolo11s_736 or yolo11n_736_scale0.2, so runs don't
    # overwrite each other.
    run_name = f"{Path(args.model).stem}_{args.imgsz}"
    if args.scale != SCALE:
        run_name += f"_scale{args.scale}"

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        epochs=EPOCHS,
        patience=PATIENCE,
        scale=args.scale,
        project=str(REPO_ROOT / "runs"),
        name=run_name,
    )
