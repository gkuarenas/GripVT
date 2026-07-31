"""Pull specific video frames out of a session's raw_video.mp4 as inspectable
PNGs, for visual debugging of events found in markers.csv (see
docs/logs/20260731_094906_burst_frame_extraction_plan.md).

Frame numbering trap (plan D1): markers.csv's `frame` column is a free-running
camera-poll counter that does NOT start at 0 -- it starts wherever the camera
happened to be when Capture Baseline ran. The indices this tool takes on the
command line are POSITIONAL INDICES into the sorted list of distinct
markers.csv `frame` values (== 0-based video frame index), NOT `frame` values
themselves. The mapping is derived from the data, never hardcoded, and the
tool aborts if the distinct-frame count doesn't match the video's own frame
count.

Output goes to <session_dir>/analysis/burst_frames/: raw (no undistortion, no
crop, no annotation) full-resolution PNGs for every requested index, a
frames.csv sidecar with per-frame numbers, and one downscaled horizontal
contact strip per range.

Reads the video sequentially with cv2.VideoCapture and does not seek with
CAP_PROP_POS_FRAMES -- frame-accurate seeking in H.264 is unreliable.

Usage:
    python tools/extract_session_frames.py --session <session_dir> --ranges 99-114,1516-1532
"""
import argparse
import csv
import statistics
import string
import sys
from pathlib import Path

import cv2

DEFAULT_MARGIN = 5


def _parse_ranges(ranges_arg: str) -> list[tuple[int, int]]:
    ranges = []
    for chunk in ranges_arg.split(","):
        lo_s, hi_s = chunk.split("-")
        lo, hi = int(lo_s), int(hi_s)
        if hi < lo:
            raise ValueError(f"Invalid range {chunk!r}: end before start.")
        ranges.append((lo, hi))
    return ranges


def _load_markers_by_frame(markers_csv: Path) -> dict[int, list[dict]]:
    """Groups markers.csv rows by the raw `frame` counter value."""
    by_frame: dict[int, list[dict]] = {}
    with open(markers_csv, newline="") as f:
        for row in csv.DictReader(f):
            by_frame.setdefault(int(row["frame"]), []).append(row)
    return by_frame


def _frame_stats(rows: list[dict], t0_ms: float) -> dict:
    """Per-frame numbers for the sidecar: relative time from timestamp_ms
    (never from video frame number / fps -- the container's fps tag is
    nominal, see output/video.py), the displacement proxy (median magnitude
    across non-autofilled markers), autofilled count, and trial_id."""
    ts_ms = float(rows[0]["timestamp_ms"])
    t_rel_s = (ts_ms - t0_ms) / 1000.0
    autofilled_flags = [row["autofilled"] == "True" for row in rows]
    n_autofilled = sum(autofilled_flags)
    non_autofilled_mags = [float(row["magnitude"]) for row, af in zip(rows, autofilled_flags) if not af]
    proxy_magnitude = statistics.median(non_autofilled_mags) if non_autofilled_mags else float("nan")
    trial_ids = {row["trial_id"] for row in rows}
    if len(trial_ids) != 1:
        raise RuntimeError(f"Frame has inconsistent trial_id across markers: {trial_ids}")
    trial_id = int(trial_ids.pop())
    return {
        "t_rel_s": t_rel_s,
        "proxy_magnitude": proxy_magnitude,
        "n_autofilled": n_autofilled,
        "trial_id": trial_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=str, required=True, help="Path to a sessionN_<ts> directory.")
    parser.add_argument(
        "--ranges", type=str, required=True,
        help="Comma-separated video-frame-index ranges, e.g. 99-114,1516-1532.",
    )
    parser.add_argument(
        "--margin", type=int, default=DEFAULT_MARGIN,
        help=f"Frames at each end of a range are labeled context (role=ctx); "
             f"the interior is the burst core (role=burst). Default {DEFAULT_MARGIN}.",
    )
    parser.add_argument(
        "--margin-note", action="store_true",
        help="Print the context-margin convention in the console summary.",
    )
    args = parser.parse_args()

    session_dir = Path(args.session)
    markers_csv = session_dir / "markers.csv"
    video_path = session_dir / "raw_video.mp4"
    if not markers_csv.exists():
        raise RuntimeError(f"markers.csv not found: {markers_csv}")
    if not video_path.exists():
        raise RuntimeError(f"raw_video.mp4 not found: {video_path}")

    ranges = _parse_ranges(args.ranges)
    if len(ranges) > 26:
        raise RuntimeError("More than 26 ranges requested; burst labels A-Z would collide.")
    burst_labels = list(string.ascii_uppercase[: len(ranges)])

    # --- D1: derive the index -> markers.csv `frame` mapping from the data. ---
    by_frame = _load_markers_by_frame(markers_csv)
    distinct_frames = sorted(by_frame.keys())
    t0_ms = float(by_frame[distinct_frames[0]][0]["timestamp_ms"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_frame_count != len(distinct_frames):
        cap.release()
        raise RuntimeError(
            f"Frame-count mismatch: video reports {video_frame_count} frames but markers.csv has "
            f"{len(distinct_frames)} distinct `frame` values. The index mapping (video_frame_index == "
            f"position in sorted distinct markers.csv frame values) is invalid for this session -- aborting "
            f"rather than producing confidently wrong output (plan D1)."
        )

    max_requested_index = max(hi for _, hi in ranges)
    if max_requested_index >= len(distinct_frames):
        cap.release()
        raise RuntimeError(
            f"Requested index {max_requested_index} is out of bounds; session has only "
            f"{len(distinct_frames)} frames (indices 0-{len(distinct_frames) - 1})."
        )

    # index -> (burst_label, role)
    index_plan: dict[int, tuple[str, str]] = {}
    for (lo, hi), label in zip(ranges, burst_labels):
        for idx in range(lo, hi + 1):
            role = "ctx" if (idx - lo < args.margin or hi - idx < args.margin) else "burst"
            index_plan[idx] = (label, role)

    out_dir = session_dir / "analysis" / "burst_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar_rows = []
    written = 0
    strip_tiles: dict[str, list] = {label: [] for label in burst_labels}

    idx = 0
    while idx <= max_requested_index:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in index_plan:
            label, role = index_plan[idx]
            csv_frame = distinct_frames[idx]
            stats = _frame_stats(by_frame[csv_frame], t0_ms)

            role_tag = "" if role == "burst" else "ctx_"
            filename = f"burst{label}_{role_tag}idx{idx:04d}_frame{csv_frame:04d}_t{stats['t_rel_s']:.2f}s.png"
            cv2.imwrite(str(out_dir / filename), frame)
            written += 1

            sidecar_rows.append({
                "burst": f"burst{label}",
                "role": role,
                "index": idx,
                "csv_frame": csv_frame,
                "t_rel_s": round(stats["t_rel_s"], 3),
                "proxy_magnitude": round(stats["proxy_magnitude"], 3),
                "n_autofilled": stats["n_autofilled"],
                "trial_id": stats["trial_id"],
            })
            strip_tiles[label].append(frame)
        idx += 1
    cap.release()

    with open(out_dir / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["burst", "role", "index", "csv_frame", "t_rel_s", "proxy_magnitude", "n_autofilled", "trial_id"]
        )
        writer.writeheader()
        writer.writerows(sidecar_rows)

    # --- D6: one downscaled horizontal contact strip per burst. ---
    STRIP_TILE_HEIGHT = 192
    for label in burst_labels:
        tiles = strip_tiles[label]
        if not tiles:
            continue
        resized = []
        for tile in tiles:
            h, w = tile.shape[:2]
            scale = STRIP_TILE_HEIGHT / h
            resized.append(cv2.resize(tile, (int(round(w * scale)), STRIP_TILE_HEIGHT)))
        strip = cv2.hconcat(resized)
        cv2.imwrite(str(out_dir / f"burst{label}_strip.png"), strip)

    # --- Console summary ---
    print(f"Session: {session_dir}")
    print(f"Video frames: {video_frame_count}  Distinct markers.csv frames: {len(distinct_frames)}  (match OK)")
    print(f"Wrote {written} PNGs + {len(burst_labels)} strip(s) + frames.csv to {out_dir}")
    if args.margin_note:
        print(f"Context margin: {args.margin} frames at each end of a range are role=ctx; interior is role=burst.")
    for label in burst_labels:
        rows_for_burst = [r for r in sidecar_rows if r["burst"] == f"burst{label}"]
        trial_ids_seen = sorted({r["trial_id"] for r in rows_for_burst})
        print(f"burst{label}: {len(rows_for_burst)} frames, trial_id(s) seen: {trial_ids_seen}")


if __name__ == "__main__":
    main()
