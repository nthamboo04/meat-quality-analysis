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
# Default: peak of the yolo11n_736 model's F1-confidence curve on the validation set.
parser.add_argument("--conf", type=float, default=0.22,
                    help="confidence threshold: boxes below this are ignored (default 0.22)")
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

# Step 3: Match predicted boxes to labelled boxes
# A predicted box is a correct detection (true positive, TP) if it has the
# same class as a labelled box and overlaps it by at least IOU_THRESHOLD.
# Each labelled box can only be matched once, so predictions are matched in
# order of confidence (most confident first). Predictions left unmatched are
# false positives (FP); labelled boxes left unmatched are misses (false
# negatives, FN).
IOU_THRESHOLD = 0.5


def iou(box_a, box_b):
    """Intersection over union of two boxes given as (x1, y1, x2, y2) pixels.

    Returns a number from 0 (no overlap) to 1 (identical boxes).
    """
    overlap_w = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    overlap_h = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = overlap_w * overlap_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def classify_false_positive(pred_box, pred_class, true_boxes, matched):
    """Work out why an unmatched predicted box is a false positive.

    Returns one of:
        "duplicate":     overlaps a labelled box of the same class that another,
                         more confident prediction already matched;
        "wrong class":   overlaps a labelled box of a different class;
        "poor location": overlaps a labelled box of the same class, but by
                         less than IOU_THRESHOLD;
        "background":    none of the above (no labelled box of the same class
                         nearby, and no strong overlap with another class).
    """
    reason = "background"
    for i, (true_class, true_box) in enumerate(true_boxes):
        overlap = iou(pred_box, true_box)
        if true_class == pred_class:
            if overlap >= IOU_THRESHOLD and matched[i]:
                return "duplicate"
            if overlap > 0:
                reason = "poor location"
        elif overlap >= IOU_THRESHOLD and reason == "background":
            reason = "wrong class"
    return reason


tp = [0] * num_classes
fp = [0] * num_classes
fn = [0] * num_classes
fp_reasons = ["duplicate", "wrong class", "poor location", "background"]
fp_by_reason = {reason: [0] * num_classes for reason in fp_reasons}

# stream=True gives results one image at a time instead of holding all of them
# in memory at once.
results = model.predict(image_paths, imgsz=args.imgsz, conf=args.conf, stream=True, verbose=False)
for image_path, result in zip(image_paths, results):
    # Labelled boxes: convert from YOLO format (class, centre x, centre y,
    # width, height as fractions of the image) to (x1, y1, x2, y2) pixels, the
    # same format as the model's predictions.
    img_height, img_width = result.orig_shape
    true_boxes = []
    label_path = DATA_DIR / args.split / "labels" / (image_path.stem + ".txt")
    for line in label_path.read_text().splitlines():
        if line.strip():
            class_id, xc, yc, w, h = line.split()
            xc, w = float(xc) * img_width, float(w) * img_width
            yc, h = float(yc) * img_height, float(h) * img_height
            true_boxes.append((int(class_id), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))

    # Predicted boxes, most confident first.
    predictions = sorted(
        zip(result.boxes.conf.tolist(), result.boxes.cls.tolist(), result.boxes.xyxy.tolist()),
        reverse=True,
    )

    matched = [False] * len(true_boxes)
    for _, pred_class, pred_box in predictions:
        pred_class = int(pred_class)
        # Find the unmatched labelled box of the same class with the largest overlap.
        best_overlap, best_index = 0.0, None
        for i, (true_class, true_box) in enumerate(true_boxes):
            if true_class == pred_class and not matched[i]:
                overlap = iou(pred_box, true_box)
                if overlap > best_overlap:
                    best_overlap, best_index = overlap, i
        if best_overlap >= IOU_THRESHOLD:
            matched[best_index] = True
            tp[pred_class] += 1
        else:
            fp[pred_class] += 1
            reason = classify_false_positive(pred_box, pred_class, true_boxes, matched)
            fp_by_reason[reason][pred_class] += 1

    for i, (true_class, _) in enumerate(true_boxes):
        if not matched[i]:
            fn[true_class] += 1

# Print the results. Precision = TP / (TP + FP): the share of predicted boxes
# that are correct. Recall = TP / (TP + FN): the share of labelled boxes found.
print()
print(f"Matching rule: same class and IoU >= {IOU_THRESHOLD}")
print(f"{'class':16s}{'labelled':>9s}{'predicted':>10s}{'TP':>5s}{'FP':>5s}{'FN':>5s}"
      f"{'precision':>11s}{'recall':>8s}")
for class_id, class_name in enumerate(class_names):
    labelled = tp[class_id] + fn[class_id]
    predicted = tp[class_id] + fp[class_id]
    precision = f"{tp[class_id] / predicted:.2f}" if predicted else "-"
    recall = f"{tp[class_id] / labelled:.2f}" if labelled else "-"
    print(f"{class_name:16s}{labelled:9d}{predicted:10d}{tp[class_id]:5d}{fp[class_id]:5d}"
          f"{fn[class_id]:5d}{precision:>11s}{recall:>8s}")
print(f"{'total':16s}{sum(tp) + sum(fn):9d}{sum(tp) + sum(fp):10d}{sum(tp):5d}{sum(fp):5d}"
      f"{sum(fn):5d}")

print()
print("Why the false positives are wrong:")
print(f"{'class':16s}" + "".join(f"{reason:>15s}" for reason in fp_reasons))
for class_id, class_name in enumerate(class_names):
    print(f"{class_name:16s}" + "".join(f"{fp_by_reason[r][class_id]:15d}" for r in fp_reasons))
