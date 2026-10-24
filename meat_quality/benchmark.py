"""Measure how long trained models take to inspect one tray, and their size.

The timings Ultralytics prints after training come from shared cloud GPUs, mix
in one-off start-up costs, and were measured on different machines, so they
cannot be compared fairly. This benchmark times every model the same way on
the same machine:

1. The images are loaded into memory first, so disk reading is not timed.
2. Each model is warmed up on a few images first, so one-off start-up costs
   (memory allocation, first-call set-up) are not timed.
3. Each model predicts every image several times, one image at a time as on a
   production line, and the time per image is recorded.

Reported per model: parameters, GFLOPs per image at the image size used,
weights file size, the median, mean and 95th-percentile time per image, the
trays per second this allows, Ultralytics' split into pre-processing,
inference and post-processing, and the achieved GFLOP/s during inference.

Run from the repository root::

    python meat_quality/benchmark.py
    python meat_quality/benchmark.py --weights runs/yolo11n_736/weights/best.pt runs/yolo11s_736/weights/best.pt
    python meat_quality/benchmark.py --weights runs/yolo11n_640/weights/best.pt --imgsz 640
"""

import argparse
import platform
import statistics
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from meat_quality.data import DATA_DIR, list_images

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = [REPO_ROOT / "runs" / "yolo11n_736_scale0.2" / "weights" / "best.pt"]

PASS_CONF = 0.10
"""Confidence threshold used while timing (the pass/fail threshold). It
affects only how many boxes post-processing handles."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Time model predictions per tray.")
    parser.add_argument("--weights", type=Path, nargs="+", default=DEFAULT_WEIGHTS,
                        help="one or more trained .pt files (default: the selected model)")
    parser.add_argument("--imgsz", type=int, default=736,
                        help="image size; must match the size the models were trained at (default 736)")
    parser.add_argument("--device", default="cpu",
                        help="'cpu', or a GPU index such as '0' (default cpu)")
    parser.add_argument("--images", type=int, default=50,
                        help="number of validation images to time on (default 50)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="times each image is predicted (default 3)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="untimed predictions before timing starts (default 10)")
    return parser.parse_args(argv)


def benchmark_model(weights: Path, images: list, imgsz: int, device: str,
                    repeats: int, warmup: int) -> dict:
    """Time a model predicting one image at a time, and measure its size.

    Args:
        weights: Path to the trained .pt file.
        images: Images already loaded into memory (BGR arrays from OpenCV).
        imgsz: Image size the model was trained at.
        device: "cpu" or a GPU index.
        repeats: Times each image is predicted.
        warmup: Untimed predictions before timing.

    Returns:
        A dict of results: "params", "gflops", "file_mb", "median_ms",
        "mean_ms", "p95_ms", "trays_per_s", "pre_ms", "infer_ms", "post_ms"
        and "gflop_per_s".

    Raises:
        ValueError: If ``images`` is empty or ``repeats`` is less than 1.
    """
    if not images or repeats < 1:
        raise ValueError("need at least one image and one repeat")
    model = YOLO(weights)
    predict = dict(imgsz=imgsz, conf=PASS_CONF, device=device, verbose=False)

    for i in range(warmup):
        model.predict(images[i % len(images)], **predict)

    totals, pre, infer, post = [], [], [], []
    for _ in range(repeats):
        for image in images:
            start = time.perf_counter()
            result = model.predict(image, **predict)[0]
            totals.append((time.perf_counter() - start) * 1000)
            # Ultralytics' own split of the time, in milliseconds.
            pre.append(result.speed["preprocess"])
            infer.append(result.speed["inference"])
            post.append(result.speed["postprocess"])

    median_ms = statistics.median(totals)
    gflops = get_flops(model.model, imgsz)
    infer_ms = statistics.mean(infer)
    return {
        "params": sum(p.numel() for p in model.model.parameters()),
        "gflops": gflops,
        "file_mb": weights.stat().st_size / 1e6,
        "median_ms": median_ms,
        "mean_ms": statistics.mean(totals),
        "p95_ms": statistics.quantiles(totals, n=20)[-1] if len(totals) >= 2 else totals[0],
        "trays_per_s": 1000 / median_ms,
        "pre_ms": statistics.mean(pre),
        "infer_ms": infer_ms,
        "post_ms": statistics.mean(post),
        # Work done per second during inference: a measure of how well the
        # hardware is used, to compare with its peak capability.
        "gflop_per_s": gflops / (infer_ms / 1000) if infer_ms > 0 else float("nan"),
    }


def model_label(weights: Path) -> str:
    """Short name for a model: its run folder, e.g. ``yolo11n_736_scale0.2``."""
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


def main(argv: list[str] | None = None):
    """Benchmark each model and print a comparison table."""
    args = parse_args(argv)
    missing = [str(w) for w in args.weights if not w.exists()]
    if missing:
        raise SystemExit("Model file(s) not found: " + ", ".join(missing))

    image_paths = list_images("valid", DATA_DIR)[:args.images]
    images = [cv2.imread(str(p)) for p in image_paths]

    device_name = (torch.cuda.get_device_name(int(args.device)) if args.device != "cpu"
                   else f"CPU: {platform.processor()}, {torch.get_num_threads()} threads")
    print(f"Device: {device_name}")
    print(f"PyTorch {torch.__version__}; {len(images)} images x {args.repeats} repeats per model, "
          f"after {args.warmup} warm-up predictions; image size {args.imgsz}")
    print()

    header = (f"{'model':24s}{'params (M)':>11s}{'GFLOPs':>8s}{'file (MB)':>10s}"
              f"{'median ms':>10s}{'p95 ms':>8s}{'trays/s':>9s}"
              f"{'pre ms':>8s}{'infer ms':>9s}{'post ms':>8s}{'GFLOP/s':>9s}")
    print(header)
    for weights in args.weights:
        r = benchmark_model(weights, images, args.imgsz, args.device, args.repeats, args.warmup)
        print(f"{model_label(weights):24s}{r['params'] / 1e6:11.2f}{r['gflops']:8.1f}"
              f"{r['file_mb']:10.1f}{r['median_ms']:10.1f}{r['p95_ms']:8.1f}"
              f"{r['trays_per_s']:9.1f}{r['pre_ms']:8.1f}{r['infer_ms']:9.1f}"
              f"{r['post_ms']:8.1f}{r['gflop_per_s']:9.1f}")


if __name__ == "__main__":
    main()
