"""Computer-vision inspection of packaged pork rasher trays.

A YOLO object detector finds and classifies packaging and product-placement
defects in 720 x 540 RGB images of trays on a conveyor, and a tray is failed
if any defect is detected.

Modules:
    data: load and explore the Roboflow dataset (labels, class counts, examples).
    metrics: match predicted boxes to labelled boxes, and tray pass/fail counts.
    train: train a YOLO model (run on a GPU, e.g. Google Colab or Kaggle).
    evaluate: evaluate a trained model on a dataset split.
    benchmark: measure prediction time, model size and FLOPs.
"""
