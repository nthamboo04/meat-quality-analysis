"""Train a YOLO object detector on the pork rasher packaging dataset.

Run from the repository root (on Colab with a GPU)::

    python meat_quality/train.py

Results (trained weights, training curves, validation metrics) are written to
runs/yolo11n_736/.
"""

from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "pork-rasher-error-packaging-4" / "data.yaml"

# Training settings
MODEL = "yolo11n.pt"  # nano model, pretrained on COCO (downloaded automatically)
IMAGE_SIZE = 736      # smallest multiple of 32 >= 720, so images are never shrunk
EPOCHS = 100          # upper limit; early stopping usually ends training sooner
PATIENCE = 20         # stop if validation mAP has not improved for 20 epochs

# The __main__ guard is needed because Ultralytics loads data in parallel
# worker processes, which re-import this file on Windows.
if __name__ == "__main__":
    model = YOLO(MODEL)
    model.train(
        data=str(DATA_YAML),
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        project=str(REPO_ROOT / "runs"),
        name="yolo11n_736",
    )
