"""Tests for meat_quality.metrics.

Each test uses small hand-made boxes whose correct answer has been worked 
out, so the tests need neither the model nor the dataset. Two classes 
are used, 0 and 1.
"""

from pathlib import Path

import pytest

from meat_quality.metrics import (ImageResult, LabelledBox, PredictedBox, TrayCounts,
                                  count_trays, f1_score, iou, match_all, tray_is_rejected)

BOX = (0, 0, 10, 10)          # a 10 x 10 box, area 100
FAR_AWAY = (100, 100, 110, 110)
THRESHOLDS = [0.5, 0.5]       # confidence threshold for classes 0 and 1

def image(labels=(), predictions=()):
    """Build an ImageResult from (class, box) labels and (confidence, class, box) predictions."""
    return ImageResult(Path("tray.jpg"),
                       [LabelledBox(*label) for label in labels],
                       [PredictedBox(*prediction) for prediction in predictions])

def test_iou():
    assert iou(BOX, BOX) == 1.0                  # identical boxes
    assert iou(BOX, FAR_AWAY) == 0.0             # no overlap
    # Overlap 5 x 10 = 50; union 100 + 100 - 50 = 150; IoU = 1/3.
    assert iou(BOX, (5, 0, 15, 10)) == pytest.approx(1/3)

def test_f1_score():
    # 2 * 9 / (2 * 9 + 16 + 25) = 18 / 59.
    assert f1_score(tp=9, fp=16, fn=25) == pytest.approx(18/59)
    assert f1_score(tp=0, fp=0, fn=0) == 0

def test_match_counts_tp_fn_and_fp():
    images = [image(labels=[(0, BOX)], predictions=[(0.9, 0, BOX)]),   # correct: TP
              image(labels=[(0, BOX)]),                                 # missed: FN
              image(predictions=[(0.9, 1, FAR_AWAY)])]                  # false alarm: FP
    counts = match_all(images, THRESHOLDS)
    assert (counts.tp, counts.fp, counts.fn) == ([1, 0], [0, 1], [1, 0])
    assert counts.fp_by_reason["background"] == [0, 1]

def test_match_ignores_predictions_below_threshold():
    # Confidence 0.3 < 0.5, so the box is ignored and the label is missed.
    counts = match_all([image(labels=[(0, BOX)], predictions=[(0.3, 0, BOX)])], THRESHOLDS)
    assert (counts.tp, counts.fp, counts.fn) == ([0, 0], [0, 0], [1, 0])

def test_match_iou_exactly_at_threshold_counts():
    # Prediction covers the top half of the label: IoU = 50 / 100 = 0.5.
    counts = match_all([image(labels=[(0, BOX)], predictions=[(0.9, 0, (0, 0, 10, 5))])],
                       THRESHOLDS)
    assert counts.tp == [1, 0]

def test_match_most_confident_prediction_is_matched_first():
    # Two labels: A = (0,0,10,10) and B = (4,0,14,10).
    # Confident box P = (1,0,11,10): IoU with A = 90/110 = 0.82, with B = 70/130 = 0.54.
    # Less confident box Q = A exactly: IoU with B = 60/140 = 0.43 (too low).
    # P goes first and takes A; Q then has nothing left, so Q is a duplicate
    # and B is missed. Matching Q first would instead give two TPs.
    label_a, label_b = (0, 0, 10, 10), (4, 0, 14, 10)
    img = image(labels=[(0, label_a), (0, label_b)],
                predictions=[(0.6, 0, label_a), (0.9, 0, (1, 0, 11, 10))])
    counts = match_all([img], THRESHOLDS)
    assert (counts.tp, counts.fp, counts.fn) == ([1, 0], [1, 0], [1, 0])

def test_match_second_box_on_same_defect_is_duplicate():
    img = image(labels=[(0, BOX)], predictions=[(0.9, 0, BOX), (0.8, 0, BOX)])
    counts = match_all([img], THRESHOLDS)
    assert (counts.tp, counts.fp) == ([1, 0], [1, 0])
    assert counts.fp_by_reason["duplicate"] == [1, 0]

def test_tray_rejected_when_a_prediction_is_exactly_at_threshold():
    assert tray_is_rejected(image(predictions=[(0.5, 0, BOX)]), THRESHOLDS)

def test_count_trays_all_four_outcomes():
    images = [image(labels=[(0, BOX)], predictions=[(0.9, 0, BOX)]),  # defective, rejected
              image(labels=[(0, BOX)]),                                # defective, passed
              image(predictions=[(0.9, 1, BOX)]),                      # good, rejected
              image()]                                                 # good, passed
    assert count_trays(images, THRESHOLDS) == TrayCounts(correct_fails=1, false_passes=1,
                                                          false_fails=1, correct_passes=1)
