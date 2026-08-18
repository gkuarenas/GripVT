"""Tests for core.roi.ROICrop.sensor_mask -- the outline + reflection-hole mask.

No tests/test_roi.py existed before this file (confirmed via grep -r ROICrop
tests/ -- every hit was a monkeypatch of .crop to the identity, never a real
sensor_mask() exercise). ROICrop.__init__ already takes a roi_path argument,
so a temp roi.json in tmp_path is enough -- no monkeypatching needed.

Shape is (h, w); polygon points are (x, y), matching cv2.fillPoly. A point is
read back from the mask as mask[y, x].
"""
import json
from pathlib import Path

from core.roi import ROICrop

SHAPE = (100, 100)
OUTLINE = [(10, 10), (90, 10), (90, 90), (10, 90)]
HOLE = [(40, 40), (60, 40), (60, 60), (40, 60)]
HOLE_2 = [(15, 65), (25, 65), (25, 75), (15, 75)]


def _write_roi_json(path: Path, mask_points=None, mask_holes=None) -> Path:
    config = {"x": 0, "y": 0, "w": 100, "h": 100, "alpha": 1.0}
    if mask_points is not None:
        config["mask_points"] = [list(p) for p in mask_points]
    if mask_holes is not None:
        config["mask_holes"] = [[list(p) for p in hole] for hole in mask_holes]
    roi_path = path / "roi.json"
    roi_path.write_text(json.dumps(config))
    return roi_path


def test_outline_with_no_holes_still_masks(tmp_path):
    roi_path = _write_roi_json(tmp_path, mask_points=OUTLINE)
    roi = ROICrop(roi_path=roi_path)

    mask = roi.sensor_mask(SHAPE)

    assert mask[50, 50] == 255
    assert mask[5, 5] == 0


def test_hole_punches_through_the_outline(tmp_path):
    roi_path = _write_roi_json(tmp_path, mask_points=OUTLINE, mask_holes=[HOLE])
    roi = ROICrop(roi_path=roi_path)

    mask = roi.sensor_mask(SHAPE)

    assert mask[50, 50] == 0
    assert mask[20, 20] == 255


def test_two_holes_both_punch_through(tmp_path):
    roi_path = _write_roi_json(tmp_path, mask_points=OUTLINE, mask_holes=[HOLE, HOLE_2])
    roi = ROICrop(roi_path=roi_path)

    mask = roi.sensor_mask(SHAPE)

    assert mask[50, 50] == 0
    assert mask[70, 20] == 0
    assert mask[20, 80] == 255


def test_degenerate_hole_is_skipped_without_raising(tmp_path):
    two_point_hole = [(1, 1), (2, 2)]
    roi_path = _write_roi_json(tmp_path, mask_points=OUTLINE, mask_holes=[two_point_hole])
    roi = ROICrop(roi_path=roi_path)

    mask = roi.sensor_mask(SHAPE)

    assert mask[50, 50] == 255
    assert mask[5, 5] == 0


def test_shape_cache_still_holds_punched_holes_on_second_call(tmp_path):
    roi_path = _write_roi_json(tmp_path, mask_points=OUTLINE, mask_holes=[HOLE])
    roi = ROICrop(roi_path=roi_path)

    first = roi.sensor_mask(SHAPE)
    second = roi.sensor_mask(SHAPE)

    assert first[50, 50] == 0
    assert second[50, 50] == 0


def test_no_mask_points_and_no_mask_holes_returns_none(tmp_path):
    roi_path = _write_roi_json(tmp_path)
    roi = ROICrop(roi_path=roi_path)

    assert roi.sensor_mask(SHAPE) is None
