"""Rebuild the per-pose calibration diagnostic figure (manuscript Figure 4.24) in the scienceplots style.

Re-runs the detector on one wearing's three stored pose images with that wearing's own winning
parameters, then draws accepted components in green and area-rejected components in red. The
pose labels follow the manuscript, not the app: the app's "max_grip" pose is written as
"Clenched fist", and "full_open" as "Fully extended".

The sensor mask comes from the current calibration/roi/roi.json. The script asserts that the
re-run count matches the count stored in the calibration record, so a changed mask cannot pass
silently.

Usage:
    python tools/fig_pose_calibration.py
    python tools/fig_pose_calibration.py --session recordings/<participant>/<session> --out fig.png
"""

import argparse
import glob
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the "science" style)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.detector import _area_accept, _labeled_components, binarize, denoise, detect, flatten_illumination  # noqa: E402
from core.roi import ROICrop  # noqa: E402

DEFAULT_SESSION = "recordings/participant009_20260901_142102/session1_20260901_142757"
DEFAULT_OUT = r"C:\Users\Guillaume\Documents\DLSU\THESIS\Writeups\Writeup (new)\Images\Figure-4.25_v2.png"
POSES = [("relaxed", "Relaxed"), ("full_open", "Fully extended"), ("max_grip", "Clenched fist")]


def pose_components(gray, params, mask):
    """Accepted detections and area-rejected component centroids, exactly as the search scores them."""
    denoised = denoise(gray, params)
    binary = binarize(flatten_illumination(denoised, params), params)
    if mask is not None:
        binary = cv2.bitwise_and(binary, mask)
    accepted = detect(denoised, binary, params)
    _, stats, centroids = _labeled_components(binary)
    accept = _area_accept(stats, params)
    rejected = [tuple(centroids[i]) for i in range(1, stats.shape[0]) if not accept[i]]
    return accepted, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    record_path = glob.glob(os.path.join(args.session, "calibration_*.json"))[0]
    record = json.load(open(record_path))
    params = record["params"]
    ts = os.path.basename(record_path)[len("calibration_"):-len(".json")]

    plt.style.use(["science", "no-latex"])
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0))
    for ax, (key, title) in zip(axes, POSES):
        gray = cv2.imread(os.path.join(args.session, f"pose_{key}_{ts}.png"), cv2.IMREAD_GRAYSCALE)
        mask = ROICrop().sensor_mask(gray.shape)
        accepted, rejected = pose_components(gray, params, mask)
        stored = record["poses"][key]["n_detected"]
        assert len(accepted) == stored, f"{key}: re-run found {len(accepted)}, record says {stored}"

        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax.scatter([d[0] for d in accepted], [d[1] for d in accepted], s=14, facecolors="none",
                   edgecolors="#2ca02c", linewidths=0.6)
        if rejected:
            ax.scatter([r[0] for r in rejected], [r[1] for r in rejected], s=14, facecolors="none",
                       edgecolors="#d62728", linewidths=0.6)
        ax.set_title(title)
        ax.set_xlabel(f"{len(accepted)}/{record['target_markers']} detected")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        print(f"{key:10s} accepted {len(accepted)}  rejected {len(rejected)}")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
