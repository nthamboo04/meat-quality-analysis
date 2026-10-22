"""Step-by-step evaluation of a trained YOLO model on the pork rasher dataset.

Run from the repository root::

    python meat_quality/evaluate.py
    python meat_quality/evaluate.py --conf 0.25 --split test

The trained weights are not stored in Git. Copy best.pt from your Google Drive
(meat-quality-runs/yolo11n_736/weights/best.pt) to the same path under runs/
in this repository, or pass a different path with --weights.
"""

# Step 1: Import libraries
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "pork-rasher-error-packaging-4"
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "yolo11n_736" / "weights" / "best.pt"

parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="path to the trained .pt file")
parser.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                    help="which split to evaluate on (default valid)")
parser.add_argument("--conf", type=float, default=0.32,
                    help="confidence threshold: boxes below this are ignored (default 0.32)")
parser.add_argument("--imgsz", type=int, default=736,
                    help="image size; must match the size the model was trained at (default 736)")
args = parser.parse_args()

# Step 2: Load the class names, the model and the images
with open(DATA_DIR / "data.yaml") as f:
    class_names = yaml.safe_load(f)["names"]
num_classes = len(class_names)

model = YOLO(args.weights)
image_paths = sorted((DATA_DIR / args.split / "images").glob("*.jpg"))
print(f"Evaluating {args.weights.name} on {len(image_paths)} {args.split} images "
      f"at confidence >= {args.conf}")

# Step 3: Run the model and count boxes per class
# For each image, count the labelled (true) boxes from its label file and the
# predicted boxes the model outputs above the confidence threshold. This does
# not yet check whether predicted boxes are in the right place; it only shows
# how many boxes the model draws compared with how many there should be.
true_counts = [0] * num_classes
pred_counts = [0] * num_classes

# stream=True gives results one image at a time instead of holding all of them
# in memory at once.
results = model.predict(image_paths, imgsz=args.imgsz, conf=args.conf, stream=True, verbose=False)
for image_path, result in zip(image_paths, results):
    label_path = DATA_DIR / args.split / "labels" / (image_path.stem + ".txt")
    for line in label_path.read_text().splitlines():
        if line.strip():
            true_counts[int(line.split()[0])] += 1
    for class_id in result.boxes.cls.tolist():
        pred_counts[int(class_id)] += 1

print()
print(f"{'class':16s}{'labelled':>10s}{'predicted':>11s}")
for class_id, class_name in enumerate(class_names):
    print(f"{class_name:16s}{true_counts[class_id]:10d}{pred_counts[class_id]:11d}")
print(f"{'total':16s}{sum(true_counts):10d}{sum(pred_counts):11d}")
