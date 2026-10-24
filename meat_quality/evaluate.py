"""Evaluate a trained YOLO model on one split of the dataset.

Reports, for the chosen split:

1. box-level TP / FP / FN, precision and recall per class at one confidence
   threshold, with the reason for each false positive;
2. the best confidence threshold for each class (highest F1), and the results
   with those thresholds;
3. tray-level pass/fail results (false passes and false fails) over a range of
   thresholds, compared with a "reject every tray" baseline;
4. a window showing the wrongly passed and failed trays at the pass/fail
   threshold.

Run from the repository root::

    python meat_quality/evaluate.py
    python meat_quality/evaluate.py --split test
    python meat_quality/evaluate.py --weights runs/yolo11n_640/weights/best.pt --imgsz 640
    python meat_quality/evaluate.py --agnostic     # class-agnostic NMS

The trained weights are not stored in Git. By default the selected model,
``runs/yolo11n_736_scale0.2/weights/best.pt``, is used; copy it there from the
training output, or pass a different path with ``--weights``.

Thresholds (per-class and pass/fail) must be chosen on the validation split;
the test split is for the final check only.
"""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ultralytics import YOLO

from meat_quality.data import (DATA_DIR, SPLITS, label_path_for, list_images, load_class_names,
                               read_labels, to_pixel_box)
from meat_quality.metrics import (FP_REASONS, ImageResult, LabelledBox, MatchCounts, PredictedBox,
                                  best_threshold_per_class, count_trays, match_all)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "yolo11n_736_scale0.2" / "weights" / "best.pt"
"""Selected model: YOLO11n at 736 px trained with scale augmentation 0.2."""

DEFAULT_CONF = 0.22
"""Single threshold: peak of the first yolo11n_736 model's F1 curve on the validation set."""

DEFAULT_PASS_CONF = 0.10
"""Pass/fail threshold, chosen from the validation sweep: the most cautious
threshold before false fails start rising (fewest defective trays let through)."""

MIN_CONF = 0.05
"""The model is run once at this confidence; higher thresholds are applied afterwards."""

IOU_THRESHOLD = 0.5
"""Minimum IoU between a predicted and a labelled box for a correct detection."""

PASS_FAIL_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)
"""Single thresholds compared in the tray pass/fail table."""

LABEL_COLOUR = "#2a78d6"       # labelled boxes: solid blue
PREDICTION_COLOUR = "#eb6834"  # predicted boxes: dashed orange


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="path to the trained .pt file")
    parser.add_argument("--split", default="valid", choices=SPLITS,
                        help="which split to evaluate on (default valid)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"confidence threshold: boxes below this are ignored "
                             f"(default {DEFAULT_CONF})")
    parser.add_argument("--imgsz", type=int, default=736,
                        help="image size; must match the size the model was trained at (default 736)")
    parser.add_argument("--agnostic", action="store_true",
                        help="use class-agnostic NMS: overlapping boxes of different classes "
                             "are also merged, keeping the most confident")
    parser.add_argument("--pass-conf", type=float, default=DEFAULT_PASS_CONF,
                        help=f"confidence threshold for tray pass/fail decisions "
                             f"(default {DEFAULT_PASS_CONF})")
    return parser.parse_args(argv)


def predict_split(model: YOLO, image_paths: list[Path], imgsz: int,
                  agnostic: bool = False) -> list[ImageResult]:
    """Run the model on every image and pair its predictions with the labels.

    The model is run once at ``MIN_CONF``. Any higher threshold can then be
    applied by discarding boxes below it, which gives the same boxes as running
    the model at that threshold: NMS keeps the most confident box of any
    overlapping group, so a low-confidence box never removes a higher one.

    Args:
        model: The trained model.
        image_paths: Images to predict on.
        imgsz: Image size the model was trained at.
        agnostic: Use class-agnostic NMS.

    Returns:
        One result per image, with labelled and predicted boxes in pixels.
    """
    images = []
    # stream=True yields one result at a time instead of holding all in memory.
    results = model.predict(image_paths, imgsz=imgsz, conf=MIN_CONF,
                            agnostic_nms=agnostic, stream=True, verbose=False)
    for image_path, result in zip(image_paths, results):
        img_height, img_width = result.orig_shape
        labels = [LabelledBox(label.class_id, to_pixel_box(label, img_width, img_height))
                  for label in read_labels(label_path_for(image_path))]
        predictions = [PredictedBox(conf, int(cls), tuple(box)) for conf, cls, box in
                       zip(result.boxes.conf.tolist(), result.boxes.cls.tolist(),
                           result.boxes.xyxy.tolist())]
        images.append(ImageResult(image_path, labels, predictions))
    return images


def print_box_results(counts: MatchCounts, class_names: list[str]):
    """Print TP/FP/FN, precision and recall per class, and the FP reasons.

    Precision = TP / (TP + FP): the share of predicted boxes that are correct.
    Recall = TP / (TP + FN): the share of labelled boxes that were found.
    """
    tp, fp, fn = counts.tp, counts.fp, counts.fn
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
        print(f"{class_name:16s}"
              + "".join(f"{counts.fp_by_reason[r][class_id]:15d}" for r in FP_REASONS))


def print_tray_results(images: list[ImageResult], per_class_thresholds: list[float],
                       num_classes: int):
    """Print tray-level pass/fail results for a range of thresholds.

    A false pass (defective tray let through) is the costly error for food
    inspection; a false fail (good tray rejected) wastes product. The "reject
    every tray" baseline shows that accuracy alone is misleading here, because
    most trays in the dataset are defective.
    """
    n_defective = sum(1 for image in images if image.labels)
    n_good = len(images) - n_defective
    print()
    print(f"=== Pass/fail per tray: {len(images)} trays, {n_defective} defective, {n_good} good ===")
    print(f"Baseline 'reject every tray': 0 false passes, {n_good} false fails, "
          f"accuracy {n_defective / len(images):.2f}")
    print()

    def print_row(label: str, thresholds: list[float]):
        trays = count_trays(images, thresholds)
        accuracy = (trays.correct_fails + trays.correct_passes) / len(images)
        print(f"{label:>22s}{trays.false_passes:>8d}/{n_defective:<4d}"
              f"{trays.false_fails:>8d}/{n_good:<4d}{accuracy:>10.2f}")

    print(f"{'thresholds':>22s}{'false passes':>13s}{'false fails':>12s}{'accuracy':>10s}")
    print_row("per-class (step 6)", per_class_thresholds)
    # From low (rejects more trays) to high (lets more trays through).
    for threshold in PASS_FAIL_SWEEP:
        print_row(f"all classes {threshold:.2f}", [threshold] * num_classes)


def find_tray_mistakes(images: list[ImageResult],
                       pass_conf: float) -> list[tuple[str, ImageResult]]:
    """Find the trays wrongly passed or failed at a single threshold.

    Args:
        images: Labelled and predicted boxes for each tray.
        pass_conf: Confidence threshold for all classes.

    Returns:
        ``(kind, image)`` pairs, where kind is "FALSE PASS" or "FALSE FAIL",
        and the image keeps only the predictions at or above ``pass_conf``.
    """
    mistakes = []
    for image in images:
        kept = [p for p in image.predictions if p.confidence >= pass_conf]
        if image.labels and not kept:
            mistakes.append(("FALSE PASS", image._replace(predictions=kept)))
        elif not image.labels and kept:
            mistakes.append(("FALSE FAIL", image._replace(predictions=kept)))
    return mistakes


def plot_tray_mistakes(mistakes: list[tuple[str, ImageResult]],
                       class_names: list[str]) -> plt.Figure:
    """Draw each wrongly passed or failed tray with its labelled and predicted boxes.

    Labelled boxes are solid blue; predicted boxes are dashed orange with their
    confidence.
    """
    n_cols = min(4, len(mistakes))
    n_rows = -(-len(mistakes) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, (kind, image) in zip(axes.flat, mistakes):
        ax.imshow(plt.imread(image.image_path))
        for class_id, (x1, y1, x2, y2) in image.labels:
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=LABEL_COLOUR,
                                           facecolor="none", linewidth=2))
            ax.text(x1, y1 - 4, class_names[class_id], color="white", fontsize=7,
                    backgroundcolor=LABEL_COLOUR)
        for conf, class_id, (x1, y1, x2, y2) in image.predictions:
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=PREDICTION_COLOUR,
                                           facecolor="none", linewidth=2, linestyle="--"))
            ax.text(x1, y2 + 14, f"{class_names[class_id]} {conf:.2f}", color="white", fontsize=7,
                    backgroundcolor=PREDICTION_COLOUR)
        ax.set_title(f"{kind}\n{image.image_path.name[:40]}", fontsize=9)

    fig.legend(handles=[patches.Patch(edgecolor=LABEL_COLOUR, facecolor="none", linewidth=2,
                                      label="labelled box"),
                        patches.Patch(edgecolor=PREDICTION_COLOUR, facecolor="none", linewidth=2,
                                      linestyle="--", label="predicted box (confidence)")],
               loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main(argv: list[str] | None = None):
    """Run the full evaluation and print the results."""
    args = parse_args(argv)
    if not args.weights.exists():
        raise SystemExit(f"Model file not found: {args.weights}\n"
                         "Copy best.pt from the training output, or pass --weights.")

    class_names = load_class_names(DATA_DIR)
    num_classes = len(class_names)
    image_paths = list_images(args.split, DATA_DIR)
    print(f"Evaluating {args.weights.name} on {len(image_paths)} {args.split} images"
          f"{' with class-agnostic NMS' if args.agnostic else ''}")

    images = predict_split(YOLO(args.weights), image_paths, args.imgsz, args.agnostic)

    # One threshold for all classes.
    print()
    print(f"=== One threshold for all classes: confidence >= {args.conf} ===")
    print(f"Matching rule: same class and IoU >= {IOU_THRESHOLD}")
    print_box_results(match_all(images, [args.conf] * num_classes, IOU_THRESHOLD), class_names)

    # A separate threshold for each class (optimistic on the split it is chosen on).
    candidates = [round(MIN_CONF + 0.01 * i, 2) for i in range(int((0.95 - MIN_CONF) / 0.01) + 1)]
    best = best_threshold_per_class(images, num_classes, args.conf, candidates, IOU_THRESHOLD)
    print()
    print("=== Best threshold for each class (highest F1) ===")
    print(f"{'class':16s}{'threshold':>10s}{'F1':>6s}")
    per_class_thresholds = []
    for class_name, best_for_class in zip(class_names, best):
        if best_for_class is None:
            # No labelled boxes of this class in the split (e.g. wrinkle):
            # F1 cannot be measured, so keep the single threshold.
            per_class_thresholds.append(args.conf)
            print(f"{class_name:16s}{'-':>10s}{'-':>6s}   (no labelled boxes; kept {args.conf})")
        else:
            threshold, f1 = best_for_class
            per_class_thresholds.append(threshold)
            print(f"{class_name:16s}{threshold:10.2f}{f1:6.2f}")

    print()
    print("=== Results with the per-class thresholds ===")
    print_box_results(match_all(images, per_class_thresholds, IOU_THRESHOLD), class_names)

    # Tray-level pass/fail.
    print_tray_results(images, per_class_thresholds, num_classes)

    # Wrongly passed or failed trays at the pass/fail threshold.
    mistakes = find_tray_mistakes(images, args.pass_conf)
    print()
    print(f"=== Wrongly passed or failed trays at confidence >= {args.pass_conf} ===")
    for kind, image in mistakes:
        print(f"{kind}: {image.image_path.name}")
    if mistakes:
        plot_tray_mistakes(mistakes, class_names)
        plt.show()


if __name__ == "__main__":
    main()
