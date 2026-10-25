"""Build the README pipeline figure (animated, light and dark) as hand-written SVGs.

The figure follows ONE marker (marker_id 49, the grid centre: row 5, col 4) through every
per-frame pipeline stage. Appearance is an idealised synthetic 9 x 11 grid of rendered Gaussian
blobs. Motion in the output strip is the cohort mean of marker 49 and of grip force (%MVC) at the 80%
target, aligned on grip_start: trials -> sessions -> participants -> cohort, equal weight per participant.

Ethics limit (URERB Form 6A): no raw or per-participant data leaves recordings/. The only output is
the SVG, which carries the cohort mean rounded to 1 decimal. The trace stays in memory.

Reads recordings/ locally, so this script runs on the author's machine only.

Usage:
    python tools/make_pipeline_figure.py
    python tools/make_pipeline_figure.py --out-dir assets   # writes -light.svg and -dark.svg
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(ROOT, "assets")

FOLLOWED_ID = 49  # row 5, col 4 of the 9-column x 11-row grid = the grid centre
FOLLOWED_RC = {"row": 5, "col": 4}
# The strip shows a 5 x 5 patch: rows 2-6 x cols 1-5 (ids row * 9 + col), with the followed marker at (5, 4).
PATCH = {r * 9 + c: (r, c) for r in range(2, 7) for c in range(1, 6)}
IDS = [FOLLOWED_ID] + [i for i in PATCH if i != FOLLOWED_ID]
OTHERS = IDS[1:]
FORCE_TARGET = 80.0  # the force bar is scaled so its cohort-mean peak meets the 80% tick
TARGET_PCT = 80
T_MIN, T_MAX, N_POINTS = -0.9, 2.3, 50  # seconds around grip_start; every 80% trial covers this
CHANNELS = ["dx", "dy", "dA"]
FORCE_COL = "pct_mvc_corr"  # grip force in % of the participant MVC; averaged but never zeroed

# ----------------------------------------------------------------------------
# 1. Aggregated trace: trial -> session -> participant -> cohort
# ----------------------------------------------------------------------------

manifest = json.load(open(os.path.join(ROOT, "notebooks", "split_manifest.json")))
t_grid = np.linspace(T_MIN, T_MAX, N_POINTS)
rest_mask = (t_grid >= T_MIN) & (t_grid <= -0.2)  # pre-grip rest window used to zero each trial

session_area = []  # baseline blob area of the 25 patch markers, per session
session_pitch = []  # median nearest-neighbour marker distance per session, px
participant_means = []  # one (76, N_POINTS) array per participant
n_trials_total = 0
n_trials_skipped = 0
n_kept = {}
n_rejected = {}
for participant, info in sorted(manifest["participants"].items()):
    pdir = glob.glob(os.path.join(ROOT, "recordings", participant + "_*"))[0]
    session_means = []
    n_kept[participant] = 0
    n_rejected[participant] = 0
    for s in info["sessions"]:  # the manifest already leaves excluded sessions out
        sdir = glob.glob(os.path.join(pdir, f"session{s}_*"))[0]
        layout = json.load(open(os.path.join(sdir, "grid_layout.json")))
        ok = layout["source"] != "fallback_sort" and layout["layout"].get(str(FOLLOWED_ID)) == FOLLOWED_RC
        for nid, (nr, nc) in PATCH.items():
            ok = ok and layout["layout"].get(str(nid)) == {"row": nr, "col": nc}
        if layout["grid_shape"] != {"n_rows": 11, "n_cols": 9}:
            ok = False
        if not ok:
            n_rejected[participant] += 1
            continue
        n_kept[participant] += 1

        # Marker pitch in the frame gate_px acts in (undistorted, rotated, full-frame px).
        # baseline_*.json holds no positions, so baseline position = x - dx, y - dy on the first frame.
        first = pd.read_csv(os.path.join(sdir, "markers.csv"), nrows=99, usecols=["marker_id", "x", "y", "dx", "dy", "area", "dA"])
        pos = np.c_[first["x"] - first["dx"], first["y"] - first["dy"]]
        dist = np.hypot(*(pos[:, None, :] - pos[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(dist, np.inf)
        session_pitch.append(np.median(dist.min(axis=1)))
        by_id = first.set_index("marker_id")
        session_area.append((by_id["area"] - by_id["dA"]).loc[IDS].to_numpy())

        cols = ["timestamp_ms", "trial_id", "target_pct"] + [f"{c}_{i}" for i in IDS for c in CHANNELS] + [FORCE_COL]
        table = pd.read_csv(os.path.join(sdir, "training_table.csv"), usecols=cols)
        events = json.load(open(os.path.join(sdir, "events.json")))["events"]
        grip_start = {e["trial_id"]: e["ts"] for e in events if e["type"] == "grip_start"}
        trials = pd.read_csv(os.path.join(sdir, "trials.csv"))
        voided = set(trials.loc[trials["voided"].astype(str) == "True", "trial_id"])

        trial_traces = []
        for trial_id, rows in table[table["target_pct"] == TARGET_PCT].groupby("trial_id"):
            if trial_id in voided or trial_id not in grip_start:
                continue
            t = (rows["timestamp_ms"].to_numpy() - grip_start[trial_id]) / 1000.0
            n_trials_total += 1
            if t.min() > T_MIN or t.max() < T_MAX:
                n_trials_skipped += 1
                continue
            trace = np.array([np.interp(t_grid, t, rows[f"{c}_{i}"].to_numpy()) for i in IDS for c in CHANNELS])
            trace = trace - trace[:, rest_mask].mean(axis=1, keepdims=True)  # zero marker motion on pre-grip rest
            ok_rows = rows[FORCE_COL].notna().to_numpy()
            force = np.interp(t_grid, t[ok_rows], rows[FORCE_COL].to_numpy()[ok_rows])  # force is not zeroed
            trace = np.vstack([trace, force])
            trial_traces.append(trace)
        if trial_traces:
            session_means.append(np.mean(trial_traces, axis=0))
    participant_means.append(np.mean(session_means, axis=0))

cohort = np.mean(participant_means, axis=0)  # (4, N_POINTS), equal weight per participant
force_raw = cohort[-1]
neighbour = {nid: cohort[3 * (j + 1):3 * (j + 2)] for j, nid in enumerate(OTHERS)}
cohort = cohort[:3]  # dx, dy, dA kept unrounded for smooth motion; numbers shown in the figure are rounded

GATE_PX = 100.0  # core/tracker.py gate_px
PITCH_PX = float(np.median(session_pitch))
print(f"marker pitch: cohort median of {len(session_pitch)} session medians = {PITCH_PX:.1f} px")
print("sessions kept     :", n_kept)
print("sessions rejected :", n_rejected)
print(f"80% trials: {n_trials_total - n_trials_skipped} used, {n_trials_skipped} skipped (window not covered)")
peaks = {}
for i, c in enumerate(CHANNELS):
    k = int(np.argmax(np.abs(cohort[i])))
    peaks[c] = (k, float(cohort[i][k]))
    print(f"peak {c}: {cohort[i][k]:+.1f} at t = {t_grid[k]:+.2f} s")
K_FORCE = int(np.argmax(force_raw))
FORCE_SCALE = FORCE_TARGET / force_raw[K_FORCE]
force_cohort = force_raw * FORCE_SCALE
print(f"force scale factor {FORCE_SCALE:.3f} (= 80 / {force_raw[K_FORCE]:.1f}); scaled rest {force_cohort[rest_mask].mean():.1f} %MVC")
print(f"force rest level (-0.9 to -0.2 s): {force_raw[rest_mask].mean():.1f} %MVC")
print(f"force peak: {force_raw[K_FORCE]:.1f} %MVC at t = {t_grid[K_FORCE]:+.2f} s")
A0 = dict(zip(IDS, np.median(session_area, axis=0)))  # each marker's own cohort-median baseline area
AREA0 = float(A0[FOLLOWED_ID])
print(f"baseline area of marker {FOLLOWED_ID}: cohort median of {len(session_area)} sessions = {AREA0:.0f} px^2; patch range {min(A0.values()):.0f}..{max(A0.values()):.0f}")
mag = np.hypot(cohort[0], cohort[1])
K_MAG = int(np.argmax(mag))
print(f"peak displacement {mag[K_MAG]:.2f} px at t = {t_grid[K_MAG]:+.2f} s")

# ----------------------------------------------------------------------------
# 2. SVG drawing helpers (plain strings)
# ----------------------------------------------------------------------------

W, H = 880, 532
MARGIN = 28
TILE = 120
GAP = (W - 2 * MARGIN - 6 * TILE) / 5
ROW1_Y, ROW2_Y, ROW3_Y = 36, 214, 386
STRIP_H = 110
PITCH = 40  # synthetic marker pitch inside a tile, px
C = TILE / 2
R2X = [(W - 3 * TILE - 2 * GAP) / 2 + k * (TILE + GAP) for k in range(3)]  # row 2: three tiles, centred

rng = np.random.default_rng(3)
parts = []


def col_x(col):
    return MARGIN + col * (TILE + GAP)


def blob(x, y, hot=False):
    return f'<use href="#{"bh" if hot else "bw"}" x="{x:.1f}" y="{y:.1f}"/>'


def lattice(span=2, pitch=PITCH, cx=C, cy=C):
    """Idealised synthetic grid points around the followed marker (i, j) = (0, 0)."""
    return [(i, j, cx + i * pitch, cy + j * pitch) for j in range(-span, span + 1) for i in range(-span, span + 1)]


def tile_open(n, col, row_y, name, x0=None, width=TILE, plot=False):
    x = col_x(col) if x0 is None else x0
    cls = "card" if plot else "tile"
    s = f'<g id="stage-{n}" class="stage">'
    s += f'<g transform="translate({x:.1f} {row_y})" clip-path="url(#tc)">' if not plot else f'<g transform="translate({x:.1f} {row_y})">'
    s += f'<rect class="{cls}" width="{width:.1f}" height="{TILE}" rx="7"/>'
    return s, x


def tile_close(n, x, row_y, name, width=TILE):
    s = "</g>"
    s += f'<rect class="frame" x="{x:.1f}" y="{row_y}" width="{width:.1f}" height="{TILE}" rx="7"/>'
    s += f'<circle class="badge" cx="{x + 11:.1f}" cy="{row_y + 11}" r="8.5"/>'
    s += f'<text class="num" x="{x + 11:.1f}" y="{row_y + 14.2}" text-anchor="middle">{n}</text>'
    s += f'<text class="lab" x="{x + width / 2:.1f}" y="{row_y + TILE + 20}" text-anchor="middle">{name}</text>'
    s += "</g>"
    return s


def arrow(x1, y1, x2, y2):
    return f'<path class="arrow" d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}" marker-end="url(#ah)"/>'


# ----------------------------------------------------------------------------
# 3. Tiles 1-6: vision row
# ----------------------------------------------------------------------------

# 1 Raw frame: barrel-curved grid lines, displaced markers, vignette, sensor noise
BARREL = 0.12  # strength of the synthetic barrel distortion (division model, stays monotone)


def barrel(px, py):
    r2 = ((px - C) / C) ** 2 + ((py - C) / C) ** 2
    k = 1 / (1 + BARREL * r2)
    return C + (px - C) * k, C + (py - C) * k


s, x = tile_open(1, 0, ROW1_Y, "Raw frame")
for k in range(-3, 4):
    p = C + (k + 0.5) * PITCH
    ts = np.linspace(-240, 360, 60)
    v_line = " ".join("%.1f,%.1f" % barrel(p, t) for t in ts)
    h_line = " ".join("%.1f,%.1f" % barrel(t, p) for t in ts)
    s += f'<polyline class="curve" points="{v_line}"/><polyline class="curve" points="{h_line}"/>'
for i, j, px, py in lattice():
    s += blob(*barrel(px, py), hot=(i, j) == (0, 0))
for _ in range(26):
    s += f'<circle class="speck" cx="{rng.uniform(2, 118):.1f}" cy="{rng.uniform(2, 118):.1f}" r="{rng.uniform(0.5, 1.2):.1f}"/>'
s += '<rect width="120" height="120" fill="url(#vig)"/>'
parts.append(s + tile_close(1, x, ROW1_Y, "Raw frame"))

# 2 Undistort: straight grid lines, regular grid
s, x = tile_open(2, 1, ROW1_Y, "Undistort")
for k in range(-3, 4):
    p = C + (k + 0.5) * PITCH
    s += f'<path class="straight" d="M{p:.1f} -4V124M-4 {p:.1f}H124"/>'
for i, j, px, py in lattice():
    s += blob(px, py, hot=(i, j) == (0, 0))
parts.append(s + tile_close(2, x, ROW1_Y, "Undistort"))

# 3 Preprocess: binarised blobs, no halo, stray noise gone
s, x = tile_open(3, 2, ROW1_Y, "Preprocess")
for i, j, px, py in lattice():
    s += f'<circle class="{"f-acc" if (i, j) == (0, 0) else "f-mk"}" cx="{px:.1f}" cy="{py:.1f}" r="8.5"/>'
parts.append(s + tile_close(3, x, ROW1_Y, "Preprocess"))

# 4 ROI mask: sensor outline keeps the grid, a stray blob outside is cut
s, x = tile_open(4, 3, ROW1_Y, "ROI mask")
for i, j, px, py in lattice():
    s += f'<circle class="{"f-acc" if (i, j) == (0, 0) else "f-mk"}" cx="{px:.1f}" cy="{py:.1f}" r="8.5"/>'
s += '<circle class="f-mut" cx="113" cy="18" r="5"/>'
s += '<path class="dim" fill-rule="evenodd" d="M-2 -2H122V122H-2Z M-8 14Q-8 6 0 6H96Q108 6 108 18V104Q108 116 96 116H0Q-8 116 -8 108Z"/>'
s += '<path class="l-acc thin" d="M-8 14Q-8 6 0 6H96Q108 6 108 18V104Q108 116 96 116H0Q-8 116 -8 108Z"/>'
parts.append(s + tile_close(4, x, ROW1_Y, "ROI mask"))

# 5 Connected components: outlines + bounding boxes
s, x = tile_open(5, 4, ROW1_Y, "Components")
for i, j, px, py in lattice():
    hot = (i, j) == (0, 0)
    s += f'<circle class="{"l-acc" if hot else "l-mk"} thin" cx="{px:.1f}" cy="{py:.1f}" r="8.5"/>'
    s += f'<rect class="{"l-acc" if hot else "l-mut"} thin" x="{px - 11:.1f}" y="{py - 11:.1f}" width="22" height="22"/>'
s += f'<rect class="f-acc" opacity="0.28" x="{C - 11}" y="{C - 11}" width="22" height="22"/>'
parts.append(s + tile_close(5, x, ROW1_Y, "Components"))

# 6 Gaussian fit: contour rings + centroid cross on the followed marker
s, x = tile_open(6, 5, ROW1_Y, "Gaussian fit")
for i, j, px, py in lattice():
    s += blob(px, py, hot=(i, j) == (0, 0))
    if (i, j) != (0, 0):
        s += f'<circle class="l-mut thin" opacity="0.5" cx="{px:.1f}" cy="{py:.1f}" r="9"/>'
for r, o in ((5, 1.0), (9, 0.75), (13, 0.5)):
    s += f'<ellipse class="l-acc ring" opacity="{o}" cx="{C}" cy="{C}" rx="{r}" ry="{r * 0.92:.1f}"/>'
s += f'<path class="l-acc cross" d="M{C - 5} {C}H{C + 5}M{C} {C - 5}V{C + 5}"/>'
parts.append(s + tile_close(6, x, ROW1_Y, "Gaussian fit"))

# ----------------------------------------------------------------------------
# 4. Tiles 7-9: tracking row, centred, flowing left to right
# ----------------------------------------------------------------------------

# 7 Kalman predict: ghost at the prior, uncertainty ellipse, last position faint
s, x = tile_open(7, 0, ROW2_Y, "Kalman predict", x0=R2X[0])
for i, j, px, py in lattice():
    if (i, j) != (0, 0):
        s += blob(px, py)
s += f'<g opacity="0.45">{blob(C - 7, C + 5, hot=True)}</g>'
s += f'<ellipse class="l-acc dash" cx="{C + 5}" cy="{C - 4}" rx="22" ry="13" transform="rotate(-28 {C + 5} {C - 4})"/>'
s += f'<circle class="l-acc ring" cx="{C + 5}" cy="{C - 4}" r="8.5" stroke-dasharray="3 2"/>'
s += f'<path class="l-acc thin" d="M{C - 7} {C + 5}L{C + 3} {C - 2}" marker-end="url(#ahA)"/>'
parts.append(s + tile_close(7, x, ROW2_Y, "Kalman predict"))

# 8 Hungarian assign: gate circle (100 px scaled to the tile) + link line, zoomed out
s, x = tile_open(8, 0, ROW2_Y, "Hungarian assign", x0=R2X[1])
sc = 0.5
s += f'<g transform="translate({C} {C}) scale({sc}) translate({-C} {-C})">'
for i, j, px, py in lattice(span=3, pitch=PITCH):
    if (i, j) != (0, 0):
        s += blob(px, py)
gate_r = GATE_PX / PITCH_PX * PITCH  # gate_px against the measured marker pitch, in tile px
s += f'<circle class="l-acc dash" cx="{C}" cy="{C}" r="{gate_r:.1f}" stroke-width="{1.6 / sc:.1f}"/>'
s += f'<circle class="l-acc" stroke-width="{1.4 / sc:.1f}" fill="none" stroke-dasharray="{3 / sc:.0f} {2 / sc:.0f}" cx="{C + 5}" cy="{C - 4}" r="8.5"/>'
s += f'<path class="l-acc" stroke-width="{1.8 / sc:.1f}" d="M{C + 5} {C - 4}L{C + 12} {C + 3}"/>'
s += blob(C + 12, C + 3, hot=True)
s += "</g>"
parts.append(s + tile_close(8, x, ROW2_Y, "Hungarian assign"))

# 9 Correct / autofill: prior, measurement, corrected state snaps toward the measurement
s, x = tile_open(9, 0, ROW2_Y, "Correct / autofill", x0=R2X[2])
for i, j, px, py in lattice():
    if (i, j) != (0, 0):
        s += blob(px, py)
s += f'<circle class="l-mut" fill="none" stroke-width="1.4" cx="{C + 12}" cy="{C + 6}" r="8.5"/>'
s += f'<circle class="l-acc ring" cx="{C - 5}" cy="{C - 6}" r="8.5" stroke-dasharray="3 2" opacity="0.7"/>'
s += f'<path class="l-acc thin" d="M{C - 5} {C - 6}L{C + 6} {C + 3}"/>'
s += f'<circle id="corrected" class="f-acc fm" cx="{C + 7}" cy="{C + 4}" r="4.5"/>'
parts.append(s + tile_close(9, x, ROW2_Y, "Correct / autofill"))

# ----------------------------------------------------------------------------
# 5. Stage 10: output strip (force bar, followed marker, live readouts)
# ----------------------------------------------------------------------------

STRIP_PITCH = 32  # spacing of the 5 x 5 patch in the strip, px
all_c = {FOLLOWED_ID: cohort, **neighbour}
PEAK_ALL = max(np.hypot(d[0], d[1]).max() for d in all_c.values())
N_MAG = int(0.6 * STRIP_PITCH // PEAK_ALL)  # one shared factor: largest peak shift stays <= 0.6 of the spacing
M_AREA = 8  # blob-area magnification: rendered area = rest area * (1 + M * dA / own baseline area)
while min((1 + M_AREA * all_c[n][2] / A0[n]).min() for n in IDS) < 0.35:
    M_AREA -= 1
S0 = 0.8  # rest blob scale (same for every marker): gradient radius 12 -> 9.6 px
R_REST = 12 * S0
print(f"patch peak |d| max {PEAK_ALL:.2f} px; motion x{N_MAG} (pitch {STRIP_PITCH}); area x{M_AREA}")

blob_scale = {n: [float(np.sqrt(1 + M_AREA * a / A0[n])) for a in all_c[n][2]] for n in IDS}
offset = {n: ((PATCH[n][1] - 3) * STRIP_PITCH, (PATCH[n][0] - 4) * STRIP_PITCH) for n in IDS}  # rest, from patch centre
pos_rel = {n: [(N_MAG * all_c[n][0][k], N_MAG * all_c[n][1][k]) for k in range(N_POINTS)] for n in IDS}
pos = {n: [(offset[n][0] + x, offset[n][1] + y) for x, y in pos_rel[n]] for n in IDS}
rad = {n: [R_REST * f for f in blob_scale[n]] for n in IDS}
trail_len, trail_ang = {}, {}
for n in IDS:
    ang_peak = float(np.degrees(np.arctan2(pos_rel[n][K_FORCE][1], pos_rel[n][K_FORCE][0])))
    trail_len[n] = [float(np.hypot(x, y)) for x, y in pos_rel[n]]
    ang = []
    for (x, y), L in zip(pos_rel[n], trail_len[n]):  # direction of the blob; the peak direction while it is still near rest
        a = float(np.degrees(np.arctan2(y, x))) if L > 1.0 else ang_peak
        ang.append(a + 360 * round((ang_peak - a) / 360))  # keep every angle within 180 deg of the peak one
    trail_ang[n] = ang
fac_all = [f * f for v in blob_scale.values() for f in v]
print(f"blob area factor range {min(fac_all):.2f}..{max(fac_all):.2f}")

# Overlap check at every sample: centre distance must exceed twice the larger blob radius of the pair.
min_dist, min_slack, req_at_min = 1e9, 1e9, 0
for k in range(N_POINTS):
    for a in range(len(IDS)):
        for b in range(a + 1, len(IDS)):
            na, nb = IDS[a], IDS[b]
            dd = float(np.hypot(pos[na][k][0] - pos[nb][k][0], pos[na][k][1] - pos[nb][k][1]))
            need = 2 * max(rad[na][k], rad[nb][k])
            if dd < min_dist:
                min_dist, req_at_min = dd, need
            min_slack = min(min_slack, dd - need)
print(f"min blob centre distance {min_dist:.1f} px (needs > {req_at_min:.1f} there); min slack over all pairs and time {min_slack:.1f} px")
assert min_slack > 0

SUP, TIMES = "px\u00b2", "\u00d7"
x_lo = min(p[0] for n in IDS for p in pos[n]) - R_REST * 1.3
x_hi = max(p[0] for n in IDS for p in pos[n]) + R_REST * 1.3
y_lo = min(p[1] for n in IDS for p in pos[n]) - R_REST * 1.3
y_hi = max(p[1] for n in IDS for p in pos[n]) + R_REST * 1.3
TAG_BAND = 18  # bottom strip of the window kept free for the magnification tag
MK_W, MK_H = x_hi - x_lo + 16, y_hi - y_lo + 16 + TAG_BAND
SX, SY, SW = MARGIN, ROW3_Y, W - 2 * MARGIN
SH = round(MK_H + 24)
H = SY + SH + 34
MK_X, MK_Y = SX + SW / 2 - MK_W / 2, SY + (SH - MK_H) / 2
CX, CY = MK_X + MK_W / 2 - (x_lo + x_hi) / 2, MK_Y + 8 - y_lo  # patch centre inside the window
print(f"strip height {SH}, marker window {MK_W:.0f} x {MK_H:.0f} px, 8 px free margin around the extents")
BAR_X, BAR_W, BAR_Y, BAR_H = 76, 216, SY + SH / 2 - 7, 14
RD_LABEL_X, RD_VAL_X, RD_UNIT_X = 604, 764, 772
RD_Y0 = SY + SH / 2 - 22


def minus(v, fmt):
    text = fmt % abs(v)
    if float(text) == 0:
        return text
    return ("−" if v < 0 else "") + text


s = f'<g id="stage-10" class="stage"><rect class="card" x="{SX}" y="{SY}" width="{SW}" height="{SH}" rx="7"/>'
s += '<g id="out-content">'

# a. grip-force bar with the 80% target tick
s += f'<rect class="track" x="{BAR_X}" y="{BAR_Y}" width="{BAR_W}" height="{BAR_H}" rx="3"/>'
s += f'<rect id="bar-fill" class="barfill" x="{BAR_X}" y="{BAR_Y}" width="{BAR_W}" height="{BAR_H}" rx="3"/>'
tick_x = BAR_X + 0.8 * BAR_W
s += f'<path class="tick-ink" d="M{tick_x:.1f} {BAR_Y - 8}V{BAR_Y + BAR_H + 8}"/>'
s += f'<text class="tick" x="{tick_x:.1f}" y="{BAR_Y - 13}" text-anchor="middle">target 80% MVC</text>'

# b. 5 x 5 patch: faint rest rings, white blobs (own motion, own area change), orange marker last
s += f'<rect class="tile" x="{MK_X:.1f}" y="{MK_Y:.1f}" width="{MK_W:.1f}" height="{MK_H:.1f}" rx="7"/>'
for n in IDS:
    s += f'<circle class="l-mut thin faint" cx="{CX + offset[n][0]:.1f}" cy="{CY + offset[n][1]:.1f}" r="{R_REST:.1f}"/>'
for n in OTHERS + [FOLLOWED_ID]:  # motion trails, rest ring centre to blob centre
    cls = "trail hot" if n == FOLLOWED_ID else "trail"
    s += f'<g transform="translate({CX + offset[n][0]:.1f} {CY + offset[n][1]:.1f})"><line id="tr-{n}" class="{cls}" x2="1"/></g>'
for n in OTHERS + [FOLLOWED_ID]:
    hot = n == FOLLOWED_ID
    cls = ' class="fm"' if hot else ""
    s += f'<g transform="translate({CX + offset[n][0]:.1f} {CY + offset[n][1]:.1f})"><g id="mk-{n}"{cls}>'
    s += f'<use href="#{"bh" if hot else "bw"}" transform="scale({S0})"/></g></g>'
s += f'<text class="tag" x="{MK_X + MK_W - 10:.1f}" y="{MK_Y + MK_H - 6:.1f}" text-anchor="end">{TIMES}{N_MAG}   area {TIMES}{M_AREA}</text>'

# c. live readouts, stepped through 8 cohort-mean values in time with the force
step_idx = [int(round(v)) for v in np.linspace(0, N_POINTS - 1, 8)]
step_idx[int(np.argmin([abs(i - K_FORCE) for i in step_idx]))] = K_FORCE
step_idx = sorted(set(step_idx))
for i, lab in enumerate(("dx", "dy", "dA")):
    ly = RD_Y0 + i * 26
    s += f'<text class="chan2" x="{RD_LABEL_X}" y="{ly}">{lab}</text>'
    s += f'<text class="unit" x="{RD_UNIT_X}" y="{ly}">{SUP if lab == "dA" else "px"}</text>'
for j, k in enumerate(step_idx):
    lines = (minus(cohort[0][k], "%.1f"), minus(cohort[1][k], "%.1f"), minus(cohort[2][k], "%.0f"))
    s += f'<text id="rd-{j}" class="rd" text-anchor="end">'
    s += "".join(f'<tspan x="{RD_VAL_X}" y="{RD_Y0 + i * 26}">{v}</tspan>' for i, v in enumerate(lines))
    s += "</text>"
s += "</g>"  # out-content
s += f'<rect class="frame" x="{SX}" y="{SY}" width="{SW}" height="{SH}" rx="7"/>'
s += f'<circle class="badge" cx="{SX + 11}" cy="{SY + 11}" r="8.5"/>'
s += f'<text class="num" x="{SX + 11}" y="{SY + 14.2}" text-anchor="middle">10</text>'
s += f'<text class="lab" x="{SX + SW / 2}" y="{SY + SH + 20}" text-anchor="middle">dx, dy, dA</text></g>'
parts.append(s)

# ----------------------------------------------------------------------------
# 6. Connectors, captions
# ----------------------------------------------------------------------------

conn = []
mid1, mid2 = ROW1_Y + TILE / 2, ROW2_Y + TILE / 2
for col in range(5):  # row 1 flows right
    conn.append(arrow(col_x(col) + TILE + 3, mid1, col_x(col + 1) - 3, mid1))
right_edge = col_x(5) + TILE
gutter_y = ROW1_Y + TILE + 36
conn.append(  # tile 6 -> right gutter -> left along the gutter between rows -> down into tile 7
    f'<path class="arrow" d="M{right_edge + 3} {mid1}H{W - 14}V{gutter_y}H{R2X[0] + TILE / 2:.1f}V{ROW2_Y - 3}" marker-end="url(#ah)"/>'
)
for k in range(2):  # row 2 flows right
    conn.append(arrow(R2X[k] + TILE + 3, mid2, R2X[k + 1] - 3, mid2))
out_x = R2X[2] + TILE + 24  # tile 9 -> down into the output strip
strip_cx = SX + SW / 2  # horizontal centre of the strip's top edge
lane_y = SY - 16  # gutter between the row 2 labels and the strip
conn.append(
    f'<path class="arrow" d="M{R2X[2] + TILE + 3:.1f} {mid2}H{out_x:.1f}V{lane_y}H{strip_cx:.1f}V{SY - 3}" marker-end="url(#ah)"/>'
)
print(f"9 -> 10 arrow tip x = {strip_cx:.1f}; strip centre x = {SX + SW / 2:.1f}")

captions = (
    f'<text class="cap" x="{MARGIN}" y="{ROW1_Y - 12}">VISION</text>'
    f'<text class="cap" x="{MARGIN}" y="{ROW2_Y - 12}">TRACKING</text>'
)

# ----------------------------------------------------------------------------
# 7. Motion (CSS keyframes only) and the two palettes
# ----------------------------------------------------------------------------

PERIOD_S = 8.0
STAGE_STEP = 4.7  # % of the loop between stage hand-offs (0.38 s)
PLAY_0, PLAY_1 = 42.0, 92.0  # % of the loop in which the real 3.2 s window plays (4.0 s: 1.25x slower)
FADE_0, FADE_1 = 96.0, 99.0
SLOWDOWN = (PLAY_1 - PLAY_0) / 100 * PERIOD_S / (T_MAX - T_MIN)
print(f"strip plays {T_MAX - T_MIN:.1f} s of real time in {(PLAY_1 - PLAY_0) / 100 * PERIOD_S:.1f} s: slowed {SLOWDOWN:.2f}x")


def at(k):  # loop percent at which sample k plays
    return PLAY_0 + (PLAY_1 - PLAY_0) * k / (N_POINTS - 1)


samples = list(range(0, N_POINTS, 2)) + [N_POINTS - 1]
OFF = "stroke:var(--muted);stroke-width:1;stroke-opacity:var(--frame)"
ON = "stroke:var(--accent);stroke-width:2.2;stroke-opacity:1"
motion = ""
for n in range(1, 10):  # tiles 1-9 light up one after the other
    a = (n - 1) * STAGE_STEP
    motion += f"@keyframes st{n}{{0%,{a:.1f}%{{{OFF}}}{a + 1.5:.1f}%,{a + 4.0:.1f}%{{{ON}}}{a + 6.0:.1f}%,100%{{{OFF}}}}}"
motion += f"@keyframes st10{{0%,{PLAY_0 - 4}%{{{OFF}}}{PLAY_0}%,{FADE_0}%{{{ON}}}{FADE_1}%,100%{{{OFF}}}}}"
motion += f"@keyframes outfade{{0%{{opacity:0}}2%,{FADE_0}%{{opacity:1}}{FADE_1}%,100%{{opacity:0}}}}"


def keyframes(name, rest, values, smp=None):
    kf = f"@keyframes {name}{{0%,{PLAY_0}%{{{rest}}}"
    kf += "".join(f"{at(k):.1f}%{{{values[k]}}}" for k in (smp or samples))
    return kf + f"{FADE_0}%,{FADE_1}%{{{values[-1]}}}100%{{{rest}}}}}"


bar_vals = [f"transform:scaleX({max(v, 0) / 100:.4f})" for v in force_cohort]
motion += keyframes("bar", bar_vals[0], bar_vals)
mk_samples = list(range(0, N_POINTS, 4)) + [N_POINTS - 1]
for n in IDS:
    vals = [f"transform:translate({x:.1f}px,{y:.1f}px) scale({sc:.3f})" for (x, y), sc in zip(pos_rel[n], blob_scale[n])]
    motion += keyframes(f"m{n}", "transform:translate(0px,0px) scale(1)", vals, mk_samples)
    tvals = [f"transform:rotate({a:.0f}deg) scale({L:.1f},1)" for a, L in zip(trail_ang[n], trail_len[n])]
    motion += keyframes(f"t{n}", f"transform:rotate({trail_ang[n][0]:.0f}deg) scale(0,1)", tvals, mk_samples)
for j, k in enumerate(step_idx):  # readouts switch with step timing, no easing
    end = at(step_idx[j + 1]) if j + 1 < len(step_idx) else FADE_1
    if j == 0:
        motion += f"@keyframes rd{j}{{0%,{end - 0.01:.2f}%{{opacity:1}}{end:.2f}%,100%{{opacity:0}}}}"
    else:
        motion += f"@keyframes rd{j}{{0%,{at(k) - 0.01:.2f}%{{opacity:0}}{at(k):.2f}%,{end - 0.01:.2f}%{{opacity:1}}{end:.2f}%,100%{{opacity:0}}}}"

motion_rules = ""
for n in range(1, 11):
    motion_rules += f"#stage-{n} .frame{{animation:st{n} {PERIOD_S}s ease-in-out infinite}}"
motion_rules += f"#out-content{{animation:outfade {PERIOD_S}s ease-in-out infinite}}"
motion_rules += f"#bar-fill{{animation:bar {PERIOD_S}s linear infinite}}"
for n in IDS:
    motion_rules += f"#mk-{n}{{animation:m{n} {PERIOD_S}s linear infinite}}#tr-{n}{{animation:t{n} {PERIOD_S}s linear infinite}}"
for j in range(len(step_idx)):
    motion_rules += f"#rd-{j}{{animation:rd{j} {PERIOD_S}s linear infinite}}"

# Static state (no animation, also the reduced-motion state): the peak-grip sample.
static_css = (
    f"#bar-fill{{transform:scaleX({force_cohort[K_FORCE] / 100:.4f})}}"
    + "".join(
        f"#mk-{n}{{transform:translate({pos_rel[n][K_FORCE][0]:.1f}px,{pos_rel[n][K_FORCE][1]:.1f}px) scale({blob_scale[n][K_FORCE]:.3f})}}"
        for n in IDS
    )
    + "".join(
        f"#tr-{n}{{transform:rotate({trail_ang[n][K_FORCE]:.0f}deg) scale({trail_len[n][K_FORCE]:.1f},1)}}"
        for n in IDS
    )
    + "".join(f"#rd-{j}{{opacity:{1 if k == K_FORCE else 0}}}" for j, k in enumerate(step_idx))
)
print("peak-grip readouts:", ascii((minus(cohort[0][K_FORCE], "%.1f"), minus(cohort[1][K_FORCE], "%.1f"), minus(cohort[2][K_FORCE], "%.0f"))))

# Palette swap only. Silicone and marker stay the same in both themes.
themes = {
    "light": {"paper": "#F7F5F0", "ink": "#1B1F24", "muted": "#8A8F98", "accent": "#E8590C", "frame": ".55"},
    "dark": {"paper": "#0D1117", "ink": "#E6EDF3", "muted": "#7D8590", "accent": "#FF7A33", "frame": ".95"},
}

style_common = """
text{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
.bg{fill:var(--paper)}
.tile{fill:var(--silicone)}
.card{fill:var(--paper)}
.badge{fill:var(--paper);stroke:var(--muted);stroke-width:1}
.num{font-size:10px;font-weight:600;fill:var(--ink)}
.lab{font-size:12.5px;font-weight:600;fill:var(--ink)}
.cap{font-size:10.5px;font-weight:600;letter-spacing:.14em;fill:var(--muted)}
.chan2{font-size:15px;font-weight:600;fill:var(--ink)}
.rd{font-size:19px;font-weight:600;fill:var(--ink);font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
.unit{font-size:12px;fill:var(--muted)}
.tick{font-size:10.5px;font-weight:600;fill:var(--muted)}
.tag{font-size:11px;font-weight:600;fill:var(--muted)}
.tick-ink{fill:none;stroke:var(--ink);stroke-width:1.6;stroke-linecap:round}
.track{fill:var(--muted);opacity:.28}
.trail{stroke:var(--marker);stroke-opacity:.5;stroke-width:1.5;stroke-linecap:round;vector-effect:non-scaling-stroke}
.trail.hot{stroke:var(--accent);stroke-opacity:1}
.barfill{fill:var(--accent);transform-box:fill-box;transform-origin:0 50%}
.faint{opacity:.45}
.dashline{fill:none;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:3 3;opacity:.8}
.arrow{fill:none;stroke:var(--muted);stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
.ah{fill:var(--muted)}
.ahA{fill:var(--accent)}
.sW{stop-color:var(--marker)}
.sH{stop-color:var(--accent)}
.sV{stop-color:#000}
.f-mk{fill:var(--marker)}
.f-acc{fill:var(--accent)}
.f-mut{fill:var(--muted)}
.l-mk{fill:none;stroke:var(--marker);stroke-width:1.2}
.l-acc{fill:none;stroke:var(--accent);stroke-width:1.6}
.l-mut{fill:none;stroke:var(--muted);stroke-width:1.2}
.thin{stroke-width:1}
.ring{stroke-width:1.4}
.cross{stroke-width:1.5;stroke-linecap:round}
.dash{stroke-dasharray:4 3}
.curve{fill:none;stroke:var(--marker);stroke-width:1;opacity:.55}
.straight{fill:none;stroke:var(--marker);stroke-width:1;opacity:.55}
.dim{fill:#000;opacity:.62}
.speck{fill:var(--marker);opacity:.18}
.frame{fill:none;stroke:var(--muted);stroke-width:1;stroke-opacity:var(--frame)}
"""

defs = f"""
<radialGradient id="gW"><stop class="sW" offset="0" stop-opacity="1"/><stop class="sW" offset=".38" stop-opacity=".85"/><stop class="sW" offset="1" stop-opacity="0"/></radialGradient>
<radialGradient id="gH"><stop class="sH" offset="0" stop-opacity="1"/><stop class="sH" offset=".55" stop-opacity="1"/><stop class="sH" offset="1" stop-opacity="0"/></radialGradient>
<radialGradient id="vig" r=".75"><stop class="sV" offset=".45" stop-opacity="0"/><stop class="sV" offset="1" stop-opacity=".7"/></radialGradient>
<circle id="bw" r="12" fill="url(#gW)"/>
<circle id="bh" r="12" fill="url(#gH)"/>
<clipPath id="tc"><rect width="{TILE}" height="{TILE}" rx="7"/></clipPath>
<marker id="ah" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="ah" d="M0 0L8 4L0 8Z"/></marker>
<marker id="ahA" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path class="ahA" d="M0 0L8 4L0 8Z"/></marker>
<marker id="ahM" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path class="ah" d="M0 0L8 4L0 8Z"/></marker>
"""

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
out_dir = ap.parse_args().out_dir
os.makedirs(out_dir, exist_ok=True)

for name, t in themes.items():
    root_vars = (
        f"svg{{--paper:{t['paper']};--ink:{t['ink']};--muted:{t['muted']};--silicone:#15171A;"
        f"--marker:#FFFFFF;--accent:{t['accent']};--frame:{t['frame']}}}"
    )
    style = root_vars + style_common + static_css + f"@media (prefers-reduced-motion:no-preference){{{motion}{motion_rules}}}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="GripVT per-frame pipeline followed through one marker">'
        f"<style>{style}</style><defs>{defs}</defs>"
        f'<rect class="bg" width="{W}" height="{H}"/>'
        f'{captions}<g id="connectors">{"".join(conn)}</g>{"".join(parts)}</svg>'
    )
    out = os.path.join(out_dir, f"gripvt-pipeline-{name}.svg")
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"saved {out}  ({len(svg.encode('utf-8')) / 1024:.1f} KB)")
