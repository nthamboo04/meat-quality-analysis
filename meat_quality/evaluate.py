"""Step-by-step evaluation of a trained YOLO model on the pork rasher dataset.

Run from the repository root::

    python meat_quality/evaluate.py
    python meat_quality/evaluate.py --conf 0.25 --split test
    python meat_quality/evaluate.py --agnostic     # class-agnostic NMS

The trained weights are not stored in Git. By default the selected model,
runs/yolo11n_736_scale0.2/weights/best.pt, is used; copy it there from the
training output, or pass a different path with --weights.
"""

# Step 1: Import libraries
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "pork-rasher-error-packaging-4"
# Selected model: YOLO11n at 736 px trained with scale augmentation 0.2.
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "yolo11n_736_scale0.2" / "weights" / "best.pt"

parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="path to the trained .pt file")
parser.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                    help="which split to evaluate on (default valid)")
# Default: peak of the first yolo11n_736 model's F1-confidence curve on the
# validation set. Step 6 also chooses a separate threshold for each class.
parser.add_argument("--conf", type=float, default=0.22,
                    help="confidence threshold: boxes below this are ignored (default 0.22)")
parser.add_argument("--imgsz", type=int, default=736,
                    help="image size; must match the size the model was trained at (default 736)")
parser.add_argument("--agnostic", action="store_true",
                    help="use class-agnostic NMS: overlapping boxes of different classes "
                         "are also merged, keeping the most confident")
args = parser.parse_args()

# Step 2: Load the class names, the model and the images
with open(DATA_DIR / "data.yaml") as f:
    class_names = yaml.safe_load(f)["names"]
num_classes = len(class_names)

model = YOLO(args.weights)
image_paths = sorted((DATA_DIR / args.split / "images").glob("*.jpg"))
print(f"Evaluating {args.weights.name} on {len(image_paths)} {args.split} images"
      f"{' with class-agnostic NMS' if args.agnostic else ''}")

# Step 3: Run the model once and store every prediction
# The model is run once at a low confidence (MIN_CONF), and any higher
# threshold is applied afterwards by discarding boxes below it. This gives the
# same boxes as running the model at that threshold, because NMS keeps the
# most confident box of any overlapping group, so a low-confidence box never
# removes a higher-confidence one. Running once makes trying many thresholds
# fast.
MIN_CONF = 0.05

images = []  # one (true_boxes, predictions) pair per image

# stream=True gives results one image at a time instead of holding all of them
# in memory at once.
results = model.predict(image_paths, imgsz=args.imgsz, conf=MIN_CONF,
                        agnostic_nms=args.agnostic, stream=True, verbose=False)
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

    # Predicted boxes as (confidence, class, box), most confident first.
    predictions = sorted(
        (conf, int(cls), box) for conf, cls, box in
        zip(result.boxes.conf.tolist(), result.boxes.cls.tolist(), result.boxes.xyxy.tolist())
    )[::-1]
    images.append((true_boxes, predictions))

# Step 4: Match predicted boxes to labelled boxes
# A predicted box is a correct detection (true positive, TP) if it has the
# same class as a labelled box and overlaps it by at least IOU_THRESHOLD.
# Each labelled box can only be matched once, so predictions are matched in
# order of confidence (most confident first). Predictions left unmatched are
# false positives (FP); labelled boxes left unmatched are misses (false
# negatives, FN).
IOU_THRESHOLD = 0.5
FP_REASONS = ["duplicate", "wrong class", "poor location", "background"]


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


def match(thresholds):
    """Count TP, FP and FN per class, ignoring predictions below a threshold.

    Args:
        thresholds: one confidence threshold per class; a predicted box of
            class c is kept only if its confidence is >= thresholds[c].

    Returns:
        (tp, fp, fn, fp_by_reason): lists of counts indexed by class, and a
        dict mapping each reason in FP_REASONS to a list of FP counts by class.
    """
    tp, fp, fn = [0] * num_classes, [0] * num_classes, [0] * num_classes
    fp_by_reason = {reason: [0] * num_classes for reason in FP_REASONS}

    for true_boxes, predictions in images:
        matched = [False] * len(true_boxes)
        for conf, pred_class, pred_box in predictions:
            if conf < thresholds[pred_class]:
                continue
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

    return tp, fp, fn, fp_by_reason


def print_results(tp, fp, fn, fp_by_reason):
    """Print TP/FP/FN, precision and recall per class, and the FP reasons.

    Precision = TP / (TP + FP): the share of predicted boxes that are correct.
    Recall = TP / (TP + FN): the share of labelled boxes that were found.
    """
    print(f"{'class':16s}{'labelled':>9s}{'predicted':>10s}{'TP':>5s}{'FP':>5s}{'FN':>5s}"
          f"{'precision':>11s}{'recall':>8s}")
    for class_id, class_name in enumerate(class_names):
        labelled = tp[class_id] + fn[class_id]
        predicted = tp[class_id] + fp[class_id]
        precision = f"{tp[class_id] / predicted:.2f}" if predicted else "-"
        recall = f"{tp[class_id] / labelled:.2f}" if labelled else "-"
        print(f"{class_name:16s}{labelled:9d}{predicted:10d}{tp[class_id]:5d}{fp[class_id]:5d}"
              f"{fn[class_id]:5d}{precision:>11s}{recall:>8s}")
    print(f"{'total':16s}{sum(tp) + sum(fn):9d}{sum(tp) + sum(fp):10d}{sum(tp):5d}"
          f"{sum(fp):5d}{sum(fn):5d}")
    print()
    print("Why the false positives are wrong:")
    print(f"{'class':16s}" + "".join(f"{reason:>15s}" for reason in FP_REASONS))
    for class_id, class_name in enumerate(class_names):
        print(f"{class_name:16s}" + "".join(f"{fp_by_reason[r][class_id]:15d}" for r in FP_REASONS))


# Step 5: Results with one threshold for all classes
print()
print(f"=== One threshold for all classes: confidence >= {args.conf} ===")
print(f"Matching rule: same class and IoU >= {IOU_THRESHOLD}")
print_results(*match([args.conf] * num_classes))

# Step 6: Choose a separate threshold for each class
# For each class, try every threshold from MIN_CONF to 0.95 and keep the one
# with the highest F1 score (the balance of precision and recall,
# F1 = 2 * TP / (2 * TP + FP + FN)). A class's threshold only affects that
# class's boxes, so each class can be tuned on its own.
#
# The thresholds are chosen on this split, so the results below are
# optimistic for this split. Choose them on the validation set and confirm
# them on the test set.
candidate_thresholds = [round(MIN_CONF + 0.01 * i, 2) for i in range(int((0.95 - MIN_CONF) / 0.01) + 1)]

best_thresholds = [args.conf] * num_classes
print()
print("=== Best threshold for each class (highest F1) ===")
print(f"{'class':16s}{'threshold':>10s}{'F1':>6s}")
for class_id, class_name in enumerate(class_names):
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        thresholds = [args.conf] * num_classes
        thresholds[class_id] = threshold
        tp, fp, fn, _ = match(thresholds)
        denominator = 2 * tp[class_id] + fp[class_id] + fn[class_id]
        f1 = 2 * tp[class_id] / denominator if denominator else 0.0
        if f1 > best_f1:
            best_f1, best_thresholds[class_id] = f1, threshold
    if tp[class_id] + fn[class_id] == 0:
        # No labelled boxes of this class in the split (e.g. wrinkle), so F1
        # cannot be measured; keep the single threshold for it.
        best_thresholds[class_id] = args.conf
        print(f"{class_name:16s}{'-':>10s}{'-':>6s}   (no labelled boxes; kept {args.conf})")
    else:
        print(f"{class_name:16s}{best_thresholds[class_id]:10.2f}{best_f1:6.2f}")

print()
print("=== Results with the per-class thresholds ===")
print_results(*match(best_thresholds))

# Step 7: Pass or fail each tray
# A tray (image) FAILS inspection if the model detects at least one defect of
# any class above that class's threshold, and PASSES otherwise. Its true
# state is DEFECTIVE if it has at least one labelled box, and GOOD if its
# label file is empty. Where the boxes are, and which class they are, does not
# matter here: any detected defect fails the tray.
#
# The two possible errors have very different costs:
#   false pass: a defective tray is let through to the customer;
#   false fail: a good tray is rejected (wasted product, or a manual re-check).


def pass_fail(thresholds):
    """Count tray-level outcomes for the given per-class thresholds.

    Returns:
        (correct_fails, false_passes, false_fails, correct_passes), where a
        "fail" is a tray the model rejects and a "pass" one it lets through.
    """
    correct_fails = false_passes = false_fails = correct_passes = 0
    for true_boxes, predictions in images:
        defective = len(true_boxes) > 0
        rejected = any(conf >= thresholds[cls] for conf, cls, _ in predictions)
        if defective and rejected:
            correct_fails += 1
        elif defective:
            false_passes += 1
        elif rejected:
            false_fails += 1
        else:
            correct_passes += 1
    return correct_fails, false_passes, false_fails, correct_passes


n_defective = sum(1 for true_boxes, _ in images if true_boxes)
n_good = len(images) - n_defective
print()
print(f"=== Pass/fail per tray: {len(images)} trays, {n_defective} defective, {n_good} good ===")

# Baseline: reject every tray. It never lets a defective tray through, but
# rejects every good one. Because most trays in this dataset are defective,
# its accuracy is already high, so accuracy alone is a misleading measure.
print(f"Baseline 'reject every tray': 0 false passes, {n_good} false fails, "
      f"accuracy {n_defective / len(images):.2f}")
print()


def print_pass_fail_row(label, thresholds):
    """Print one row of the pass/fail table for the given thresholds."""
    correct_fails, false_passes, false_fails, correct_passes = pass_fail(thresholds)
    accuracy = (correct_fails + correct_passes) / len(images)
    print(f"{label:>22s}{false_passes:>8d}/{n_defective:<4d}{false_fails:>8d}/{n_good:<4d}"
          f"{accuracy:>10.2f}")


print(f"{'thresholds':>22s}{'false passes':>13s}{'false fails':>12s}{'accuracy':>10s}")
print_pass_fail_row("per-class (step 6)", best_thresholds)
# One threshold for every class, from low (rejects more trays, fewer false
# passes) to high (lets more trays through, fewer false fails).
for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
    print_pass_fail_row(f"all classes {threshold:.2f}", [threshold] * num_classes)
