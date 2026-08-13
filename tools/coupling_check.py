"""Measure how well a wearing transferred grip force to the sensing surface.

Answers a question that session-level correlation alone cannot: when r is low, is it because
the sleeve failed to couple, or because `max_disp` latched onto a marker that sits offset from
its stored baseline and never responds to grip?

Reports, per session:

  frame_motion  median per-frame movement of the whole marker field. Flat across good and bad
                sessions alike -- if this is normal, the markers are still moving.
  rest / grip   median max displacement while force is below 15 % / above 60 % of the trial
                peak, taken per trial and pooled. Both are measured inside the same trial, so
                a per-session offset cancels.
  ratio         grip / rest. THE coupling measure. Good wearings give 2.3-4.0x. Participant
                003's failed wearings give 1.18-1.21x.
  r             Pearson r of max_disp against force, as recorded.
  r_reref       the same, after re-referencing every marker to its own session's resting
                median position instead of the stored Capture Baseline.
  rigid         median whole-grid translation -- the sleeve sliding as a unit.

Reading the output:

  ratio low  AND r_reref no better than r   -> genuine coupling failure (p003 s6, s7)
  ratio ok   AND r_reref much better than r -> a max_disp artifact, not a bad wearing (p004 s10)

Use the RATIO to flag wearings, never an absolute displacement floor: a floor fails a
large-forearm participant on every wearing simply because their signal is small, which is a
finding about the participant rather than a defect in the mount.

Usage:
    python tools/coupling_check.py recordings/participant003_20260812_132754
    python tools/coupling_check.py recordings/participant00*  --shift-sweep
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

GRIP_HI = 0.60   # fraction of the trial peak that counts as "gripping"
GRIP_LO = 0.15   # fraction of the trial peak that counts as "at rest inside the trial"
MIN_PEAK_KG = 2.0  # trials below this are rest trials; they have no grip window


def _channel(df, prefix):
    """Marker columns for one channel, ordered by marker_id rather than by string sort."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda s: int(s.split("_")[1]))


def _pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return np.corrcoef(a[ok], b[ok])[0, 1]


def _grip_ratio(maxd, force, trial_id):
    """Median displacement while gripping vs while resting, compared within each trial."""
    rest, grip = [], []
    for t in np.unique(trial_id):
        m = trial_id == t
        f, d = force[m], maxd[m]
        if not np.isfinite(f).any():
            continue
        peak = np.nanmax(f)
        if peak < MIN_PEAK_KG:
            continue
        hi, lo = f > GRIP_HI * peak, f < GRIP_LO * peak
        if hi.sum() < 3 or lo.sum() < 3:
            continue
        rest.append(np.nanmedian(d[lo]))
        grip.append(np.nanmedian(d[hi]))
    if not rest:
        return np.nan, np.nan
    return np.nanmedian(rest), np.nanmedian(grip)


def _best_shift(maxd, force, t_s, span=3.0, step=0.05):
    """Largest r reachable by sliding force in time. Separates a sync error from a real loss."""
    best_r, best_shift = -np.inf, 0.0
    ok = np.isfinite(maxd) & np.isfinite(force) & np.isfinite(t_s)
    d, f, t = maxd[ok], force[ok], t_s[ok]
    for shift in np.arange(-span, span + step / 2, step):
        r = _pearson(d, np.interp(t + shift, t, f))
        if r > best_r:
            best_r, best_shift = r, shift
    return best_r, best_shift


def analyse_session(session_dir, shift_sweep=False):
    table = os.path.join(session_dir, "training_table.csv")
    if not os.path.exists(table):
        return None
    df = pd.read_csv(table)

    dx = df[_channel(df, "dx_")].to_numpy()
    dy = df[_channel(df, "dy_")].to_numpy()
    force = df["force_kg_corr"].to_numpy()
    trial_id = df["trial_id"].to_numpy()
    is_rest_trial = (df["target_pct"] == 0).to_numpy()
    t_s = (df["timestamp_ms"].to_numpy() - df["timestamp_ms"].iloc[0]) / 1000.0

    disp = np.hypot(dx, dy)
    maxd = np.nanmax(disp, axis=1)

    # Re-reference: subtract each marker's own resting median instead of the stored baseline.
    bx = np.nanmedian(dx[is_rest_trial], axis=0)
    by = np.nanmedian(dy[is_rest_trial], axis=0)
    maxd_reref = np.nanmax(np.hypot(dx - bx, dy - by), axis=1)

    rest, grip = _grip_ratio(maxd, force, trial_id)
    row = {
        "session": int(re.search(r"session(\d+)_", os.path.basename(session_dir)).group(1)),
        "frame_motion": np.nanmedian(np.abs(np.diff(disp, axis=0)).mean(axis=1)),
        "rest": rest,
        "grip": grip,
        "ratio": grip / rest if rest else np.nan,
        "r": _pearson(maxd, force),
        "r_reref": _pearson(maxd_reref, force),
        "rigid": np.nanmedian(np.hypot(np.nanmean(dx, axis=1), np.nanmean(dy, axis=1))),
    }
    if shift_sweep:
        row["r_best_shift"], row["shift_s"] = _best_shift(maxd, force, t_s)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("participant_dirs", nargs="+")
    ap.add_argument("--shift-sweep", action="store_true",
                    help="also sweep the force signal +/-3 s to rule out a sync error (slower)")
    args = ap.parse_args()

    for participant in args.participant_dirs:
        sessions = sorted(glob.glob(os.path.join(participant, "session*")),
                          key=lambda p: int(re.search(r"session(\d+)_", os.path.basename(p)).group(1)))
        rows = [r for r in (analyse_session(s, args.shift_sweep) for s in sessions) if r]
        if not rows:
            print(f"{participant}: no training_table.csv found")
            continue
        print(f"\n== {os.path.basename(participant)} ==")
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
