"""Headless tests for output/video.py's VideoLogWriter (Phase 2 of
docs/logs/20260730_110247_raw_video_log_plan.md). Pure tmp_path-based:
no Tk, no camera, synthetic frames only.
"""
import os

import cv2
import numpy as np
import pytest

from output.video import VideoLogWriter, VIDEO_EXT, VIDEO_FPS

SIZE = (1024, 768)  # (width, height), matches cv2.VideoWriter's expected order


def make_frames(n: int) -> list[np.ndarray]:
    return [np.random.randint(0, 256, (768, 1024, 3), dtype=np.uint8) for _ in range(n)]


def test_write_and_close_produces_nonempty_file(tmp_path):
    path = str(tmp_path / f"raw_video{VIDEO_EXT}")
    writer = VideoLogWriter(path, size=SIZE, fps=VIDEO_FPS)
    frames = make_frames(10)
    for f in frames:
        writer.write(f)
    writer.close()

    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    assert writer.frame_count == 10
    assert writer.path == path


def test_close_is_idempotent(tmp_path):
    path = str(tmp_path / f"raw_video{VIDEO_EXT}")
    writer = VideoLogWriter(path, size=SIZE, fps=VIDEO_FPS)
    writer.write(make_frames(1)[0])
    writer.close()
    writer.close()  # must not raise


def test_reread_matches_written_frame_count_and_shape(tmp_path):
    path = str(tmp_path / f"raw_video{VIDEO_EXT}")
    writer = VideoLogWriter(path, size=SIZE, fps=VIDEO_FPS)
    frames = make_frames(15)
    for f in frames:
        writer.write(f)
    writer.close()

    cap = cv2.VideoCapture(path)
    decoded = 0
    last_shape = None
    for _ in range(len(frames) + 5):  # bounded: a stuck backend must not hang the suite
        ok, f = cap.read()
        if not ok:
            break
        decoded += 1
        last_shape = f.shape
    cap.release()

    assert decoded == len(frames)
    assert last_shape == frames[0].shape


def test_init_raises_on_unwritable_path(tmp_path):
    bad_path = str(tmp_path / "does_not_exist" / f"raw_video{VIDEO_EXT}")
    with pytest.raises(Exception):
        VideoLogWriter(bad_path, size=SIZE, fps=VIDEO_FPS)


def test_write_tracks_first_and_last_timestamp(tmp_path):
    path = str(tmp_path / f"raw_video{VIDEO_EXT}")
    writer = VideoLogWriter(path, size=SIZE, fps=VIDEO_FPS)
    for frame, ts in zip(make_frames(3), [100.0, 200.0, 350.0]):
        writer.write(frame, ts)
    writer.close()

    assert writer.first_ts_ms == 100.0
    assert writer.last_ts_ms == 350.0


def test_write_without_timestamp_leaves_ts_none(tmp_path):
    path = str(tmp_path / f"raw_video{VIDEO_EXT}")
    writer = VideoLogWriter(path, size=SIZE, fps=VIDEO_FPS)
    writer.write(make_frames(1)[0])  # existing callers don't pass timestamp_ms
    writer.close()

    assert writer.first_ts_ms is None
    assert writer.last_ts_ms is None
