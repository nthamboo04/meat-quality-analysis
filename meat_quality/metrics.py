"""Compare predicted boxes with labelled boxes, per box and per tray.

Box-level matching follows the usual object-detection rule: a predicted box is
a correct detection if it has the same class as a labelled box and overlaps it 
by at least an IoU threshold. Each labelled box can be matched only once, so 
predictions are matched most confident first. Unmatched predictions are false 
positives and unmatched labelled boxes are misses.

A tray fails inspection if any predicted box is above its class's confidence 
threshold, and is truly defective if it has at least one labelled box.

All boxes are (x1, y1, x2, y2) in pixels.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from meat_quality.data import Box

FP_REASONS = ("duplicate", "wrong class", "poor location", "background")
"""Categories of false positive, as returned by ``classify_false_positive``."""

class LabelledBox(NamedTuple):
    """A labelled (true) box."""

    class_id: int
    box: Box

class PredictedBox(NamedTuple):
    """A box predicted by the model."""

    confidence: float
    class_id: int
    box: Box

class ImageResult(NamedTuple):
    """The labelled and predicted boxes for one image."""

    image_path: Path
    labels: list[LabelledBox]
    predictions: list[PredictedBox]

@dataclass
class MatchCounts:
    """Box-level counts per class, indexed by class id."""

    tp: list[int]
    fp: list[int]
    fn: list[int]
    fp_by_reason: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def zeros(cls, num_classes):
        """All-zero counts for ``num_classes`` classes."""
        return cls([0] * num_classes, [0] * num_classes, [0] * num_classes,
                   {reason: [0] * num_classes for reason in FP_REASONS})

class TrayCounts(NamedTuple):
    """Tray-level outcomes: "fail" = rejected by the model, "pass" = let through."""

    correct_fails: int
    false_passes: int   # defective tray let through
    false_fails: int    # good tray rejected
    correct_passes: int

def iou(box_a, box_b):
    """Intersection over union of two boxes.

    Args:
        box_a: First box, (x1, y1, x2, y2).
        box_b: Second box, (x1, y1, x2, y2).

    Returns:
        Overlap area divided by combined area: 0 for no overlap, 1 for
        identical boxes. Returns 0 if both boxes have zero area.
    """
    overlap_w = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    overlap_h = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = overlap_w * overlap_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0

def classify_false_positive(prediction, labels, matched, iou_threshold=0.5):
    """Work out why an unmatched predicted box is a false positive.

    Args:
        prediction: The unmatched predicted box.
        labels: All labelled boxes in the image.
        matched: ``matched[i]`` is True if ``labels[i]`` was already matched
            by a more confident prediction.
        iou_threshold: IoU needed for a match.

    Returns:
        One of ``FP_REASONS``:
        "duplicate" if it overlaps (IoU >= threshold) an already-matched label
        of the same class. "poor location" if it overlaps a label of the same
        class, but by less than the threshold. "wrong class" if it overlaps a
        label of another class by at least the threshold. Otherwise "background".
    """
    reason = "background"
    for label, is_matched in zip(labels, matched):
        overlap = iou(prediction.box, label.box)
        if label.class_id == prediction.class_id:
            if overlap >= iou_threshold and is_matched:
                return "duplicate"
            if overlap > 0:
                reason = "poor location"
        elif overlap >= iou_threshold and reason == "background":
            reason = "wrong class"
    return reason

def match_image(image, thresholds, counts, iou_threshold=0.5):
    """Match one image's predictions to its labels and add to ``counts``.

    Args:
        image: The image's labelled and predicted boxes.
        thresholds: Confidence threshold per class; predicted boxes of class c
            below ``thresholds[c]`` are ignored.
        counts: Running totals, updated in place.
        iou_threshold: IoU needed for a match.
    """
    matched = [False] * len(image.labels)
    for prediction in sorted(image.predictions, reverse=True):  # most confident first
        if prediction.confidence < thresholds[prediction.class_id]:
            continue
        # The unmatched label of the same class with the largest overlap.
        best_overlap, best_index = 0.0, None
        for i, label in enumerate(image.labels):
            if label.class_id == prediction.class_id and not matched[i]:
                overlap = iou(prediction.box, label.box)
                if overlap > best_overlap:
                    best_overlap, best_index = overlap, i
        if best_overlap >= iou_threshold:
            matched[best_index] = True
            counts.tp[prediction.class_id] += 1
        else:
            counts.fp[prediction.class_id] += 1
            reason = classify_false_positive(prediction, image.labels, matched, iou_threshold)
            counts.fp_by_reason[reason][prediction.class_id] += 1

    for label, is_matched in zip(image.labels, matched):
        if not is_matched:
            counts.fn[label.class_id] += 1

def match_all(images, thresholds, iou_threshold=0.5):
    """Box-level TP, FP and FN per class over a set of images.

    Args:
        images: Labelled and predicted boxes for each image.
        thresholds: Confidence threshold per class.
        iou_threshold: IoU needed for a match.

    Returns:
        Totals over all images.
    """
    counts = MatchCounts.zeros(len(thresholds))
    for image in images:
        match_image(image, thresholds, counts, iou_threshold)
    return counts

def f1_score(tp, fp, fn):
    """F1 score, the balance of precision and recall: 2TP / (2TP + FP + FN).

    Returns 0 when there are no labelled or predicted boxes (TP + FP + FN = 0).
    """
    denominator = 2 * tp + fp + fn
    if denominator != 0:
        return 2 * tp / denominator
    else:
        return 0

def best_threshold_per_class(images, num_classes, default, candidates, iou_threshold=0.5):
    """Choose, for each class, the confidence threshold with the highest F1.

    A class's threshold only affects that class's boxes, so each class is
    tuned separately. Ties go to the lowest threshold (fewer misses).

    The thresholds should be chosen on the validation set; evaluated on the
    same images, their results are optimistic.

    Args:
        images: Labelled and predicted boxes for each image.
        num_classes: Number of classes.
        default: Threshold used for the other classes while one is tuned.
        candidates: Thresholds to try, in increasing order.
        iou_threshold: IoU needed for a match.

    Returns:
        For each class, ``(best threshold, its F1)``, or None for a class with
        no labelled boxes.
    """
    best = []
    for class_id in range(num_classes):
        best_for_class = None
        for threshold in candidates:
            thresholds = [default] * num_classes
            thresholds[class_id] = threshold
            counts = match_all(images, thresholds, iou_threshold)
            if counts.tp[class_id] + counts.fn[class_id] == 0:
                break
            f1 = f1_score(counts.tp[class_id], counts.fp[class_id], counts.fn[class_id])
            if best_for_class is None or f1 > best_for_class[1]:
                best_for_class = (threshold, f1)
        best.append(best_for_class)
    return best

def tray_is_rejected(image, thresholds):
    """Whether the model fails a tray: any predicted box above its class's threshold."""
    return any(p.confidence >= thresholds[p.class_id] for p in image.predictions)

def count_trays(images, thresholds):
    """Tray-level pass/fail outcomes for the given per-class thresholds.

    A tray is defective if it has at least one labelled box.

    Args:
        images: Labelled and predicted boxes for each image (tray).
        thresholds: Confidence threshold per class.

    Returns:
        Counts of correct fails, false passes, false fails and correct passes.
    """
    correct_fails = false_passes = false_fails = correct_passes = 0
    for image in images:
        defective = bool(image.labels)
        rejected = tray_is_rejected(image, thresholds)
        if defective and rejected:
            correct_fails += 1
        elif defective:
            false_passes += 1
        elif rejected:
            false_fails += 1
        else:
            correct_passes += 1
    return TrayCounts(correct_fails, false_passes, false_fails, correct_passes)
