"""Raw video log writer (docs/logs/20260730_110247_raw_video_log_plan.md).

Branch A won Phase 1's codec/backend probe: on this machine's OpenCV build,
cv2.VideoWriter's default backend resolves to MSMF (not FFMPEG) for .mp4
writes, and MSMF's native H.264 encoder MFT opens and encodes real H.264
with zero extra dependencies -- no ffmpeg install needed (see the plan's
"Phase 1 results" section, run 2026-07-30). Explicitly passing
cv2.CAP_FFMPEG is broken on this machine (no OpenH264 DLL shipped with
opencv-python) and must not be done.

Known ceiling: cv2.VideoWriter's Python API exposes no CRF/quality
parameter. Quality is whatever MSMF's encoder defaults to -- this is
documented, not worked around. If it proves inadequate, Branch B (an
ffmpeg subprocess with explicit -crf) is the spec'd fallback, not built here.
"""
import cv2
import numpy as np

VIDEO_FPS = 30.0  # matches the probe's FPS and D3's full-30fps decision
VIDEO_EXT = ".mp4"  # Phase 1 probe: only .mp4 opened under MSMF; .avi failed for every fourcc
VIDEO_FOURCC = "avc1"  # Phase 1 probe: avc1/H264/h264/X264 were byte-identical under MSMF; avc1 chosen as most conventional


class VideoLogWriter:
    def __init__(self, path: str, size: tuple[int, int], fps: float = VIDEO_FPS) -> None:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"VideoLogWriter: failed to open {path!r} (fourcc={VIDEO_FOURCC!r}, size={size}, fps={fps})")
        self.path = path
        self.frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
