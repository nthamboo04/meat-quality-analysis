"""Load and summarise a YOLO-format object-detection dataset.

The dataset is expected in the layout produced by a Roboflow YOLO export::

    <root>/
        data.yaml              class names (``names``) and count (``nc``)
        train/images/*.jpg     one label file per image, same stem
        train/labels/*.txt
        valid/...
        test/...

Each label file holds one line per bounding box::

    <class_id> <x_centre> <y_centre> <width> <height>

where ``class_id`` is a zero-based integer index into the class names and the
four box values are normalised to the image size, i.e. lie in ``[0, 1]``. An
empty label file means the image contains no labelled objects (here: no
defects).

Run as a module to print a summary of the downloaded dataset::

    python -m meat_quality.dataset [path/to/dataset]
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Default dataset location, relative to the repository root.
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "pork-rasher-error-packaging-4"

#: Split folder names used by the Roboflow export.
SPLITS = ("train", "valid", "test")

#: Image file extensions accepted when scanning a split.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

#: Separator Roboflow inserts between the original filename and the hash it
#: appends to each exported (possibly augmented) copy.
ROBOFLOW_SEPARATOR = ".rf."


@dataclass(frozen=True)
class BoundingBox:
    """One labelled object in normalised YOLO coordinates.

    Attributes:
        class_id: Zero-based class index.
        x_centre: Box centre x, as a fraction of image width, in [0, 1].
        y_centre: Box centre y, as a fraction of image height, in [0, 1].
        width: Box width as a fraction of image width, in [0, 1].
        height: Box height as a fraction of image height, in [0, 1].
    """

    class_id: int
    x_centre: float
    y_centre: float
    width: float
    height: float


@dataclass
class ImageRecord:
    """An image file and the boxes labelled on it.

    Attributes:
        image_path: Path to the image file.
        boxes: Labelled boxes; empty if the image has no labelled objects.
    """

    image_path: Path
    boxes: list[BoundingBox] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        """Identifier of the original photo this image was derived from."""
        return source_id(self.image_path.name)


def parse_label_line(line: str) -> BoundingBox:
    """Parse one line of a YOLO label file.

    Args:
        line: A line of the form ``"<class_id> <x> <y> <w> <h>"``.

    Returns:
        The parsed bounding box.

    Raises:
        ValueError: If the line does not have exactly five fields, the class
            id is not a non-negative integer, a coordinate is not a number,
            or a coordinate lies outside [0, 1].
    """
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"expected 5 fields, got {len(fields)}: {line!r}")

    try:
        class_id = int(fields[0])
    except ValueError:
        raise ValueError(f"class id is not an integer: {line!r}") from None
    if class_id < 0:
        raise ValueError(f"class id is negative: {line!r}")

    try:
        coords = [float(v) for v in fields[1:]]
    except ValueError:
        raise ValueError(f"coordinate is not a number: {line!r}") from None
    if any(not 0.0 <= c <= 1.0 for c in coords):
        raise ValueError(f"coordinate outside [0, 1]: {line!r}")

    return BoundingBox(class_id, *coords)


def read_label_file(path: Path) -> list[BoundingBox]:
    """Read all bounding boxes from a YOLO label file.

    Blank lines are ignored, and a missing final newline is handled (Roboflow
    exports omit it).

    Args:
        path: Path to the ``.txt`` label file.

    Returns:
        The boxes in file order; empty if the file has no boxes.

    Raises:
        ValueError: If any line is malformed; the message names the file.
    """
    boxes = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_label_line(line))
        except ValueError as err:
            raise ValueError(f"{path}:{line_no}: {err}") from None
    return boxes


def source_id(filename: str) -> str:
    """Return the original-photo identifier for a Roboflow export filename.

    Roboflow names exported files ``<original>.rf.<hash>.<ext>``, so all
    augmented copies of one photo share the part before ``.rf.``. Filenames
    without that marker are returned without their extension.

    Args:
        filename: Image or label filename (not a full path).

    Returns:
        The shared identifier, e.g. ``"frame_with_red_10049_jpg"``.
    """
    if ROBOFLOW_SEPARATOR in filename:
        return filename.split(ROBOFLOW_SEPARATOR)[0]
    return Path(filename).stem


def load_class_names(root: Path) -> list[str]:
    """Read class names from ``data.yaml`` in the dataset root.

    Args:
        root: Dataset root folder.

    Returns:
        Class names, indexed by class id.

    Raises:
        FileNotFoundError: If ``data.yaml`` does not exist.
        ValueError: If ``names`` is missing or disagrees with ``nc``.
    """
    with open(root / "data.yaml") as f:
        config = yaml.safe_load(f)
    names = config.get("names")
    if not isinstance(names, list):
        raise ValueError("data.yaml has no 'names' list")
    if "nc" in config and config["nc"] != len(names):
        raise ValueError(f"data.yaml: nc={config['nc']} but {len(names)} names")
    return names


def load_split(root: Path, split: str) -> list[ImageRecord]:
    """Load every image in a split together with its labels.

    Args:
        root: Dataset root folder.
        split: Split name, e.g. ``"train"``.

    Returns:
        One record per image, sorted by filename.

    Raises:
        FileNotFoundError: If the split's ``images`` folder does not exist,
            or an image has no matching label file.
        ValueError: If a label file is malformed.
    """
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"no images folder: {image_dir}")

    records = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"no label file for {image_path.name}")
        records.append(ImageRecord(image_path, read_label_file(label_path)))
    return records


def class_counts(records: list[ImageRecord], n_classes: int) -> dict[str, list[int]]:
    """Count boxes and images per class.

    Args:
        records: Images to count over.
        n_classes: Number of classes; class ids must be below this.

    Returns:
        ``{"boxes": [...], "images": [...]}``, each a list indexed by class
        id. ``boxes`` counts every box; ``images`` counts images containing
        at least one box of that class.

    Raises:
        ValueError: If a record contains a class id ``>= n_classes``.
    """
    boxes = [0] * n_classes
    images = [0] * n_classes
    for record in records:
        ids = [box.class_id for box in record.boxes]
        for class_id in ids:
            if class_id >= n_classes:
                raise ValueError(f"class id {class_id} >= {n_classes} in {record.image_path.name}")
            boxes[class_id] += 1
        for class_id in set(ids):
            images[class_id] += 1
    return {"boxes": boxes, "images": images}


def find_source_overlap(
    split_a: list[ImageRecord], split_b: list[ImageRecord]
) -> set[str]:
    """Find original photos that appear in both splits.

    An overlap means a model evaluated on ``split_b`` has already seen a copy
    of the same photo during training on ``split_a`` (data leakage).

    Args:
        split_a: Records of the first split.
        split_b: Records of the second split.

    Returns:
        Source identifiers present in both splits.
    """
    return {r.source_id for r in split_a} & {r.source_id for r in split_b}


def summarise(root: Path = DEFAULT_ROOT) -> str:
    """Build a plain-text summary of the dataset.

    Args:
        root: Dataset root folder.

    Returns:
        A multi-line report of image, box and class counts per split, the
        number of distinct source photos, and source overlap between splits.
    """
    names = load_class_names(root)
    splits = {s: load_split(root, s) for s in SPLITS}
    lines = []

    for split, records in splits.items():
        counts = class_counts(records, len(names))
        n_empty = sum(1 for r in records if not r.boxes)
        n_sources = len({r.source_id for r in records})
        lines.append(
            f"{split}: {len(records)} images from {n_sources} source photos, "
            f"{n_empty} with no labels"
        )
        for class_id, name in enumerate(names):
            lines.append(
                f"  {name:16s} boxes={counts['boxes'][class_id]:5d}  "
                f"images={counts['images'][class_id]:5d}"
            )

    lines.append("source photos shared between splits:")
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlap = find_source_overlap(splits[a], splits[b])
        lines.append(f"  {a}/{b}: {len(overlap)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarise(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT))
