"""Throwaway dev tool: decide Phase 1's Branch A vs Branch B fork for the raw
video log (docs/logs/20260730_110247_raw_video_log_plan.md).

For each candidate (fourcc, backend, container) combination, opens a
cv2.VideoWriter into a tempfile.mkdtemp() path, writes the test frames,
closes it, then re-opens with cv2.VideoCapture and reads the frames back.
An opened writer that produced an undecodable (or short) file counts as a
FAIL, not a pass -- OpenCV silently no-ops unsupported fourccs on some
builds, so "isOpened() was True" alone proves nothing.

Test footage: this machine has no GripVT camera attached. Per the plan and
per Guillaume's instruction, substitutes the 13 real (not synthetic) camera
captures under
calibration/intrinsics/results/20260701_174717_calibration/raw_images/,
looped to reach the target frame count. This is checkerboard footage from
the fisheye intrinsics calibration, not real forearm/marker footage -- it
exercises the codec/container plumbing faithfully (real sensor noise, real
JPEG/PNG artifacts) but is NOT a substitute for the compression-ratio
measurement the plan's D3 sizing table depends on. That measurement still
needs a real GripVT session on the rig (Phase 7).

If no native combination round-trips as genuine H.264, and ffmpeg is on
PATH, also probes Branch B (piping raw frames to an ffmpeg subprocess).

Usage:
    python tools/probe_video_codec.py
"""
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
IMAGE_DIR = ROOT / "calibration" / "intrinsics" / "results" / "20260701_174717_calibration" / "raw_images"
TARGET_FRAME_COUNT = 900  # ~30 s at 30 fps, per the plan
FPS = 30.0

# (fourcc, backend, backend_name, container_ext)
CANDIDATES = [
    (fourcc, backend, backend_name, ext)
    for fourcc in ["avc1", "H264", "h264", "X264"]
    for backend, backend_name in [(cv2.CAP_ANY, "default"), (cv2.CAP_FFMPEG, "CAP_FFMPEG")]
    for ext in [".mp4", ".avi"]
]


def load_test_frames() -> list[np.ndarray]:
    """Load the 13 real checkerboard captures and loop them to
    TARGET_FRAME_COUNT. Logs actual resolution read via cv2.imread rather
    than assuming (1024, 768); resizes only if a frame doesn't match the
    first frame's shape."""
    paths = sorted(IMAGE_DIR.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No source images found in {IMAGE_DIR}")
    base_frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"cv2.imread failed to decode {p}")
        base_frames.append(img)
    shape0 = base_frames[0].shape
    print(f"Loaded {len(base_frames)} source images from {IMAGE_DIR.name}/, first frame shape={shape0}")
    for i, f in enumerate(base_frames):
        if f.shape != shape0:
            print(f"  resizing frame {i} from {f.shape} to match {shape0}")
            base_frames[i] = cv2.resize(f, (shape0[1], shape0[0]))

    frames = list(itertools.islice(itertools.cycle(base_frames), TARGET_FRAME_COUNT))
    print(f"Looped to {len(frames)} frames for the probe (target {TARGET_FRAME_COUNT}).")
    return frames


def probe_native(frames: list[np.ndarray], tmp_dir: str) -> list[dict]:
    h, w = frames[0].shape[:2]
    size = (w, h)
    results = []
    for fourcc_str, backend, backend_name, ext in CANDIDATES:
        label = f"{fourcc_str}/{backend_name}/{ext}"
        out_path = os.path.join(tmp_dir, f"probe_{fourcc_str}_{backend_name}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        result = {
            "label": label,
            "fourcc": fourcc_str,
            "backend": backend_name,
            "ext": ext,
            "opened": False,
            "writer_backend": None,
            "bytes": 0,
            "bytes_per_frame": 0.0,
            "decoded_frames": 0,
            "decoded_shape": None,
            "codec_tag": None,
            "pass": False,
        }

        if backend == cv2.CAP_ANY:
            writer = cv2.VideoWriter(out_path, fourcc, FPS, size)
        else:
            writer = cv2.VideoWriter(out_path, backend, fourcc, FPS, size)

        result["opened"] = writer.isOpened()
        try:
            result["writer_backend"] = writer.getBackendName()
        except Exception:
            result["writer_backend"] = None
        if result["opened"]:
            for f in frames:
                writer.write(f)
        writer.release()

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            results.append(result)
            print(f"  {label}: opened={result['opened']} writer_backend={result['writer_backend']} -> no/empty output file")
            continue

        result["bytes"] = os.path.getsize(out_path)
        result["bytes_per_frame"] = result["bytes"] / len(frames)

        cap = cv2.VideoCapture(out_path)
        codec_tag = int(cap.get(cv2.CAP_PROP_FOURCC))
        result["codec_tag"] = (
            chr(codec_tag & 0xFF) + chr((codec_tag >> 8) & 0xFF) + chr((codec_tag >> 16) & 0xFF) + chr((codec_tag >> 24) & 0xFF)
        )
        decoded = 0
        last_shape = None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            decoded += 1
            last_shape = f.shape
        cap.release()
        result["decoded_frames"] = decoded
        result["decoded_shape"] = last_shape

        result["pass"] = (
            result["opened"]
            and decoded == len(frames)
            and last_shape == frames[0].shape
            and result["codec_tag"].strip("\x00").lower() in {"avc1", "h264", "x264"}
        )

        print(
            f"  {label}: opened={result['opened']} writer_backend={result['writer_backend']} "
            f"bytes={result['bytes']} bytes/frame={result['bytes_per_frame']:.1f} "
            f"decoded={decoded}/{len(frames)} shape={last_shape} "
            f"codec_tag={result['codec_tag']!r} pass={result['pass']}"
        )
        results.append(result)
    return results


def probe_ffmpeg(frames: list[np.ndarray], tmp_dir: str) -> dict:
    h, w = frames[0].shape[:2]
    out_path = os.path.join(tmp_dir, "probe_ffmpeg.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(int(FPS)),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        out_path,
    ]
    log_path = os.path.join(tmp_dir, "ffmpeg.log")
    with open(log_path, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log_f, stderr=log_f)
        assert proc.stdin is not None
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait(timeout=120)

    result = {
        "label": "ffmpeg/libx264/.mp4",
        "opened": proc.returncode == 0,
        "bytes": 0,
        "bytes_per_frame": 0.0,
        "decoded_frames": 0,
        "decoded_shape": None,
        "codec_tag": None,
        "pass": False,
    }
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        print(f"  ffmpeg: returncode={proc.returncode} -> no/empty output file (see {log_path})")
        return result

    result["bytes"] = os.path.getsize(out_path)
    result["bytes_per_frame"] = result["bytes"] / len(frames)

    cap = cv2.VideoCapture(out_path)
    codec_tag = int(cap.get(cv2.CAP_PROP_FOURCC))
    result["codec_tag"] = (
        chr(codec_tag & 0xFF) + chr((codec_tag >> 8) & 0xFF) + chr((codec_tag >> 16) & 0xFF) + chr((codec_tag >> 24) & 0xFF)
    )
    decoded = 0
    last_shape = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        decoded += 1
        last_shape = f.shape
    cap.release()
    result["decoded_frames"] = decoded
    result["decoded_shape"] = last_shape
    result["pass"] = decoded == len(frames) and last_shape == frames[0].shape

    print(
        f"  ffmpeg: returncode={proc.returncode} bytes={result['bytes']} "
        f"bytes/frame={result['bytes_per_frame']:.1f} decoded={decoded}/{len(frames)} "
        f"shape={last_shape} codec_tag={result['codec_tag']!r} pass={result['pass']}"
    )
    return result


def probe_write_latency(frames: list[np.ndarray], tmp_dir: str) -> None:
    """Phase 5: times VideoLogWriter.write() in isolation (no camera, no Tk
    poll loop) over the same looped frame set, to check the per-frame encode
    cost against POLL_MS=33 (ui/session/camera_feed.py) as a real number
    instead of inferring it from feed smoothness. Uses the actual
    output.video.VideoLogWriter class -- the real Branch A winner config --
    not a bare cv2.VideoWriter."""
    sys.path.insert(0, str(ROOT))
    from output.video import VideoLogWriter

    h, w = frames[0].shape[:2]
    out_path = os.path.join(tmp_dir, "probe_write_latency.mp4")
    writer = VideoLogWriter(out_path, size=(w, h))
    latencies_ms = []
    for f in frames:
        t0 = time.perf_counter()
        writer.write(f)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    writer.close()

    latencies_ms.sort()
    n = len(latencies_ms)
    p50 = latencies_ms[int(n * 0.50)]
    p95 = latencies_ms[int(n * 0.95)]
    p_max = latencies_ms[-1]
    budget_ms = 33.0
    margin_ms = budget_ms - p95
    print(f"\n--- VideoLogWriter.write() latency, {n} frames, isolated (no camera/UI) ---")
    print(f"  p50={p50:.2f}ms  p95={p95:.2f}ms  max={p_max:.2f}ms  (POLL_MS budget = {budget_ms:.0f}ms)")
    print(f"  p95 margin vs {budget_ms:.0f}ms budget: {margin_ms:.2f}ms "
          f"{'(fits, with margin)' if margin_ms > 0 else '(EXCEEDS BUDGET on this machine)'}")
    print("  NOTE: this machine's CPU/disk may not match the actual rig laptop -- "
          "this is informative, not the final verification. The on-rig before/after "
          "frames-per-second A/B measurement is deferred to Phase 7.")


def extrapolate(bytes_per_frame: float) -> None:
    session_frames = 65 * 60 * FPS  # 65 min, per D3
    session_bytes = bytes_per_frame * session_frames
    print(f"  -> {bytes_per_frame:.1f} bytes/frame extrapolates to "
          f"{session_bytes / (1024**3):.3f} GiB / 65-min session, "
          f"{session_bytes * 100 / (1024**3):.1f} GiB / 100 sessions")
    print("  NOTE: extrapolated from checkerboard stand-in footage (near-static, "
          "high-contrast), not real forearm/marker footage. Real compression ratio "
          "on GripVT session footage is still to be confirmed on the rig (Phase 7).")


def main() -> None:
    print(f"OpenCV {cv2.__version__}")
    ffmpeg_path = shutil.which("ffmpeg")
    print(f"ffmpeg on PATH: {ffmpeg_path}")

    frames = load_test_frames()
    tmp_dir = tempfile.mkdtemp(prefix="gripvt_probe_")
    print(f"Writing probe outputs to {tmp_dir}")

    print("\n--- Branch A: native cv2.VideoWriter ---")
    native_results = probe_native(frames, tmp_dir)
    winners = [r for r in native_results if r["pass"]]

    if winners:
        best = winners[0]
        print(f"\nBranch A WINNER: {best['label']}")
        extrapolate(best["bytes_per_frame"])
        probe_write_latency(frames, tmp_dir)
    else:
        print("\nNo native combination produced a genuinely re-readable H.264 file.")
        if ffmpeg_path:
            print("\n--- Branch B: ffmpeg subprocess (libx264) ---")
            ff_result = probe_ffmpeg(frames, tmp_dir)
            if ff_result["pass"]:
                print("\nBranch B WINNER: ffmpeg/libx264/.mp4")
                extrapolate(ff_result["bytes_per_frame"])
            else:
                print("\nBranch B probe did not round-trip correctly either.")
        else:
            print("ffmpeg not found on PATH. Not installing it automatically per instructions.")
            print("BLOCKED: no native H.264 and no ffmpeg available -- report to Guillaume.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal probe time: {time.time() - start:.1f}s")
