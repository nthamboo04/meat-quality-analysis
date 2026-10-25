"""Tests for meat_quality.data, using hand-made label lines and files."""

import pytest

from meat_quality.data import Label, parse_label_line, read_labels, to_pixel_box

def test_parse_label_line():
    assert parse_label_line("2 0.5 0.25 0.1 0.2") == Label(2, 0.5, 0.25, 0.1, 0.2)
    for bad_line in ["1 0.5 0.5 0.1",          # too few values
                     "1 0.5 abc 0.1 0.1",      # not a number
                     "1 1.2 0.5 0.1 0.1"]:     # number outside 0-1
        with pytest.raises(ValueError):
            parse_label_line(bad_line)

def test_read_labels_without_final_newline(tmp_path):
    # Roboflow label files have no final newline. Both boxes must still be read.
    label_file = tmp_path / "tray.txt"
    label_file.write_text("0 0.1 0.1 0.1 0.1\n1 0.9 0.9 0.1 0.1")
    assert [label.class_id for label in read_labels(label_file)] == [0, 1]

def test_to_pixel_box():
    # Centre (0.5, 0.5), size 0.2 x 0.4 in a 100 x 50 image.
    # Centre (50, 25), size 20 x 20 -> corners (40, 15) to (60, 35).
    assert to_pixel_box(Label(0, 0.5, 0.5, 0.2, 0.4), 100, 50) == pytest.approx((40, 15, 60, 35))
