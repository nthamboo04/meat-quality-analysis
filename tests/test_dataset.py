"""Tests for meat_quality.dataset, using small hand-made datasets."""

from pathlib import Path

import pytest

from meat_quality.dataset import (
    BoundingBox,
    ImageRecord,
    class_counts,
    find_source_overlap,
    load_class_names,
    load_split,
    parse_label_line,
    read_label_file,
    source_id,
)


def make_dataset(root: Path, labels: dict[str, dict[str, str]]) -> Path:
    """Write a tiny dataset: ``labels[split][filename_stem] = label text``."""
    (root / "data.yaml").write_text("nc: 3\nnames: ['a', 'b', 'c']\n")
    for split, files in labels.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, text in files.items():
            (root / split / "images" / f"{stem}.jpg").write_bytes(b"")
            (root / split / "labels" / f"{stem}.txt").write_text(text)
    return root


# --- parse_label_line -------------------------------------------------------

def test_parse_label_line_valid():
    box = parse_label_line("2 0.5 0.25 0.1 0.2")
    assert box == BoundingBox(2, 0.5, 0.25, 0.1, 0.2)


def test_parse_label_line_accepts_boundary_values():
    assert parse_label_line("0 0 1 0 1") == BoundingBox(0, 0.0, 1.0, 0.0, 1.0)


@pytest.mark.parametrize(
    "line",
    [
        "1 0.5 0.5 0.1",           # too few fields
        "1 0.5 0.5 0.1 0.1 0.9",   # too many fields (e.g. a polygon label)
        "x 0.5 0.5 0.1 0.1",       # class id not an integer
        "1.5 0.5 0.5 0.1 0.1",     # class id not an integer
        "-1 0.5 0.5 0.1 0.1",      # negative class id
        "1 0.5 abc 0.1 0.1",       # coordinate not a number
        "1 1.2 0.5 0.1 0.1",       # coordinate above 1
        "1 0.5 -0.1 0.1 0.1",      # coordinate below 0
    ],
)
def test_parse_label_line_rejects_invalid(line):
    with pytest.raises(ValueError):
        parse_label_line(line)


# --- read_label_file --------------------------------------------------------

def test_read_label_file_without_trailing_newline(tmp_path):
    # Roboflow omits the final newline; naive concatenation of such files
    # merges lines, so check both boxes are still read.
    path = tmp_path / "img.txt"
    path.write_text("0 0.1 0.1 0.1 0.1\n1 0.9 0.9 0.1 0.1")
    assert [b.class_id for b in read_label_file(path)] == [0, 1]


def test_read_label_file_empty_and_blank_lines(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    blanks = tmp_path / "blanks.txt"
    blanks.write_text("\n0 0.5 0.5 0.1 0.1\n\n")
    assert read_label_file(empty) == []
    assert len(read_label_file(blanks)) == 1


def test_read_label_file_error_names_file_and_line(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 0.5 0.1 0.1\n0 0.5 0.5\n")
    with pytest.raises(ValueError, match=r"bad\.txt:2"):
        read_label_file(path)


# --- source_id --------------------------------------------------------------

def test_source_id_groups_roboflow_copies():
    a = source_id("frame_with_red_10049_jpg.rf.cd786c4f.jpg")
    b = source_id("frame_with_red_10049_jpg.rf.0a1b2c3d.jpg")
    assert a == b == "frame_with_red_10049_jpg"


def test_source_id_without_roboflow_marker():
    assert source_id("photo.jpg") == "photo"


# --- load_class_names / load_split ------------------------------------------

def test_load_class_names(tmp_path):
    make_dataset(tmp_path, {})
    assert load_class_names(tmp_path) == ["a", "b", "c"]


def test_load_class_names_rejects_nc_mismatch(tmp_path):
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['a', 'b', 'c']\n")
    with pytest.raises(ValueError):
        load_class_names(tmp_path)


def test_load_split_pairs_images_with_labels(tmp_path):
    make_dataset(tmp_path, {"train": {"img1": "0 0.5 0.5 0.1 0.1", "img2": ""}})
    records = load_split(tmp_path, "train")
    assert [r.image_path.name for r in records] == ["img1.jpg", "img2.jpg"]
    assert [len(r.boxes) for r in records] == [1, 0]


def test_load_split_missing_label_file(tmp_path):
    make_dataset(tmp_path, {"train": {"img1": ""}})
    (tmp_path / "train" / "images" / "orphan.jpg").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="orphan.jpg"):
        load_split(tmp_path, "train")


def test_load_split_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "train")


# --- class_counts -----------------------------------------------------------

def test_class_counts_boxes_and_images():
    records = [
        ImageRecord(Path("1.jpg"), [BoundingBox(0, .5, .5, .1, .1), BoundingBox(0, .2, .2, .1, .1)]),
        ImageRecord(Path("2.jpg"), [BoundingBox(0, .5, .5, .1, .1), BoundingBox(2, .5, .5, .1, .1)]),
        ImageRecord(Path("3.jpg"), []),
    ]
    counts = class_counts(records, n_classes=3)
    assert counts["boxes"] == [3, 0, 1]   # every box counted
    assert counts["images"] == [2, 0, 1]  # each image counted once per class


def test_class_counts_rejects_unknown_class():
    records = [ImageRecord(Path("1.jpg"), [BoundingBox(5, .5, .5, .1, .1)])]
    with pytest.raises(ValueError):
        class_counts(records, n_classes=3)


# --- find_source_overlap ----------------------------------------------------

def test_find_source_overlap():
    train = [ImageRecord(Path("p1.rf.aa.jpg")), ImageRecord(Path("p2.rf.bb.jpg"))]
    test = [ImageRecord(Path("p2.rf.cc.jpg")), ImageRecord(Path("p3.rf.dd.jpg"))]
    assert find_source_overlap(train, test) == {"p2"}
