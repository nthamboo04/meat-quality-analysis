"""Load and explore the pork rasher packaging dataset.

The dataset is the Roboflow "Pork Rasher Error(Packaging)" v4 export in YOLO
format, stored in ``data/pork-rasher-error-packaging-4/``::

    data.yaml              class names ("names") and class count ("nc")
    train/images/*.jpg     720 x 540 RGB images
    train/labels/*.txt     one label file per image, with the same file name
    valid/...
    test/...

Each line of a label file is one labelled box::

    <class_id> <x_centre> <y_centre> <width> <height>

where ``class_id`` is a zero-based index into the class names and the four box
values are fractions of the image width and height, between 0 and 1. An empty
label file means the tray has no labelled defect.

Run as a script to print the class counts and show a label-count chart::

    python meat_quality/data.py
"""

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import yaml

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "pork-rasher-error-packaging-4"
"""Default dataset folder, found from this file's location so scripts work from any folder."""

SPLITS = ("train", "valid", "test")
"""Names of the predefined dataset splits."""

Box = tuple[float, float, float, float]
"""A box as (x1, y1, x2, y2) in pixels: top-left and bottom-right corners."""


class Label(NamedTuple):
    """One line of a YOLO label file, in normalised coordinates (0 to 1)."""

    class_id: int
    x_centre: float
    y_centre: float
    width: float
    height: float


def load_class_names(data_dir: Path = DATA_DIR) -> list[str]:
    """Read the class names from the dataset's data.yaml.

    Args:
        data_dir: Dataset folder containing data.yaml.

    Returns:
        Class names, indexed by class id.

    Raises:
        FileNotFoundError: If data.yaml does not exist.
        ValueError: If data.yaml has no list of names, or nc does not match
            the number of names.
    """
    with open(data_dir / "data.yaml") as f:
        config = yaml.safe_load(f)
    names = config.get("names")
    if not isinstance(names, list):
        raise ValueError(f"{data_dir / 'data.yaml'} has no 'names' list")
    if "nc" in config and config["nc"] != len(names):
        raise ValueError(f"data.yaml: nc={config['nc']} but {len(names)} names are listed")
    return names


def list_images(split: str, data_dir: Path = DATA_DIR) -> list[Path]:
    """List a split's image files, sorted by name.

    Args:
        split: One of SPLITS.
        data_dir: Dataset folder.

    Returns:
        Paths of the ``.jpg`` images in ``<data_dir>/<split>/images``.

    Raises:
        ValueError: If split is not one of SPLITS.
        FileNotFoundError: If the split's image folder does not exist.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    image_dir = data_dir / split / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image folder not found: {image_dir}")
    return sorted(image_dir.glob("*.jpg"))


def label_path_for(image_path: Path) -> Path:
    """Return the label file that belongs to an image.

    Images live in ``<split>/images/`` and their labels in ``<split>/labels/``,
    with the same file name and a ``.txt`` extension.
    """
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def parse_label_line(line: str) -> Label:
    """Parse one line of a YOLO label file.

    Args:
        line: A line of the form ``"<class_id> <x_centre> <y_centre> <width> <height>"``.

    Returns:
        The parsed label.

    Raises:
        ValueError: If the line does not have five fields, the class id is not
            a non-negative integer, or a box value is not a number in [0, 1].
    """
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"expected 5 values, got {len(fields)}: {line!r}")
    try:
        class_id = int(fields[0])
        values = [float(v) for v in fields[1:]]
    except ValueError:
        raise ValueError(f"not a valid label line: {line!r}") from None
    if class_id < 0:
        raise ValueError(f"negative class id: {line!r}")
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"box value outside [0, 1]: {line!r}")
    return Label(class_id, *values)


def read_labels(label_path: Path) -> list[Label]:
    """Read every labelled box in a label file.

    Blank lines are skipped. A missing final newline (as in the Roboflow
    export) is handled.

    Args:
        label_path: Path to a YOLO ``.txt`` label file.

    Returns:
        The labels in file order; empty if the tray has no labelled defect.

    Raises:
        FileNotFoundError: If the label file does not exist.
        ValueError: If a line is malformed; the message names the file and line.
    """
    labels = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            labels.append(parse_label_line(line))
        except ValueError as err:
            raise ValueError(f"{label_path}, line {line_number}: {err}") from None
    return labels


def to_pixel_box(label: Label, img_width: int, img_height: int) -> Box:
    """Convert a normalised YOLO label to a pixel box.

    YOLO stores the box centre and size as fractions of the image; plotting and
    overlap calculations need the corners in pixels.

    Args:
        label: The label to convert.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        The box as (x1, y1, x2, y2) in pixels.
    """
    x_centre, width = label.x_centre * img_width, label.width * img_width
    y_centre, height = label.y_centre * img_height, label.height * img_height
    return (x_centre - width / 2, y_centre - height / 2,
            x_centre + width / 2, y_centre + height / 2)


def count_labels(split: str, num_classes: int, data_dir: Path = DATA_DIR) -> tuple[list[int], int]:
    """Count the labelled boxes of each class in a split.

    Args:
        split: One of ``SPLITS``.
        num_classes: Number of classes; every class id must be below this.
        data_dir: Dataset folder.

    Returns:
        ``(box_counts, n_no_defect)``: the number of boxes of each class
        (indexed by class id), and the number of images with no boxes.

    Raises:
        ValueError: If a label file is malformed or has a class id
            ``>= num_classes``.
    """
    box_counts = [0] * num_classes
    n_no_defect = 0
    for image_path in list_images(split, data_dir):
        labels = read_labels(label_path_for(image_path))
        if not labels:
            n_no_defect += 1
        for label in labels:
            if label.class_id >= num_classes:
                raise ValueError(f"class id {label.class_id} >= {num_classes} in "
                                 f"{label_path_for(image_path)}")
            box_counts[label.class_id] += 1
    return box_counts, n_no_defect


def plot_label_counts(class_names: list[str], counts: dict[str, tuple[list[int], int]],
                      image_counts: dict[str, int]) -> plt.Figure:
    """Draw one bar chart of label counts per split.

    Each split gets its own y-axis because the splits are very different
    sizes (train is about 40 times bigger than test).

    Args:
        class_names: Class names, indexed by class id.
        counts: ``counts[split]`` is the ``(box_counts, n_no_defect)`` result
            of ``count_labels`` for that split.
        image_counts: Number of images in each split, for the panel titles.

    Returns:
        The figure, with one panel per split.
    """
    bar_labels = class_names + ["(no defect)"]
    bar_colours = ["#2a78d6"] * len(class_names) + ["#a3a29b"]  # defects blue, no defect grey

    fig, axes = plt.subplots(1, len(counts), figsize=(15, 4.5))
    for ax, (split, (box_counts, n_no_defect)) in zip(axes, counts.items()):
        bars = ax.bar(bar_labels, box_counts + [n_no_defect], color=bar_colours)
        ax.bar_label(bars, padding=2)
        ax.set_title(f"{split} ({image_counts[split]} images)")
        ax.set_ylabel("number of boxes / images")
        ax.tick_params(axis="x", rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    """Print dataset statistics and show label-count charts."""
    class_names = load_class_names()
    num_classes = len(class_names)
    print("class_names:", class_names)
    print("num_classes:", num_classes)

    image_counts = {split: len(list_images(split)) for split in SPLITS}
    for split, n_images in image_counts.items():
        print(f"{split}: {n_images} images")

    counts = {split: count_labels(split, num_classes) for split in SPLITS}
    print()
    print(f"{'class':16s}" + "".join(f"{split:>8s}" for split in SPLITS))
    for class_id, class_name in enumerate(class_names):
        print(f"{class_name:16s}" + "".join(f"{counts[split][0][class_id]:8d}" for split in SPLITS))
    print(f"{'(no defect)':16s}" + "".join(f"{counts[split][1]:8d}" for split in SPLITS))

    plot_label_counts(class_names, counts, image_counts)
    plt.show()

if __name__ == "__main__":
    main()