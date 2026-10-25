"""Reproduce thesis Table 5.8, the signal dynamic range per participant.

For every session in notebooks/split_manifest.json:
  frame peak  = largest marker magnitude in a frame (markers.csv)
  trial peak  = largest frame peak inside one trial_id
Per participant:
  rest  = median trial peak over target_pct == 0 trials
  grip  = median trial peak over target_pct == 80 trials
  range = grip - rest, ratio = grip / rest   (all in px)
  med5/20/50 = median trial peak at the middle levels (graded-response check)
  iqr0       = interquartile range of the rest trial peaks (noise reference)
  range_iqr0 = range / iqr0
  p80_gt_0   = P(80% trial peak > rest trial peak), Mann-Whitney U / (n80 * n0)
  rho        = Spearman correlation of target level against trial peak

Usage (from the repo root):
    python tools/dynamic_range.py
Writes docs/results/dynamic_range_table58.csv.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'results', 'dynamic_range_table58.csv')

with open(os.path.join(ROOT, 'notebooks', 'split_manifest.json')) as f:
    manifest = json.load(f)['participants']

rows = []
for pid in sorted(manifest):
    num = int(pid.replace('participant', ''))
    peaks = []                                        # (target_pct, trial peak) over all sessions
    for s in manifest[pid]['sessions']:
        folder = glob.glob(os.path.join(ROOT, 'recordings', f'participant{num:03d}_*', f'session{s}_*'))
        assert len(folder) == 1, f'expected one folder for p{num} s{s}, got {folder}'
        mk = pd.read_csv(os.path.join(folder[0], 'markers.csv'), usecols=['frame', 'magnitude', 'trial_id'])
        tr = pd.read_csv(os.path.join(folder[0], 'trials.csv'), usecols=['trial_id', 'target_pct'])
        mk = mk.dropna(subset=['trial_id'])
        frame_peak = mk.groupby(['trial_id', 'frame'])['magnitude'].max()
        trial_peak = frame_peak.groupby('trial_id').max()
        target = tr.set_index('trial_id')['target_pct']
        for tid, pk in trial_peak.items():
            if tid in target.index:
                peaks.append((target[tid], pk))
    pk = pd.DataFrame(peaks, columns=['target_pct', 'peak'])
    r0, r80 = pk.loc[pk.target_pct == 0, 'peak'], pk.loc[pk.target_pct == 80, 'peak']
    rest, grip = r0.median(), r80.median()
    med = pk.groupby('target_pct')['peak'].median()
    iqr0 = r0.quantile(0.75) - r0.quantile(0.25)
    u = stats.mannwhitneyu(r80, r0, alternative='two-sided').statistic
    rows.append({'participant': pid, 'rest_px': rest, 'grip80_px': grip, 'range_px': grip - rest,
                 'ratio': grip / rest, 'n0': len(r0), 'n80': len(r80),
                 'med5_px': med[5], 'med20_px': med[20], 'med50_px': med[50],
                 'iqr0_px': iqr0, 'range_iqr0': (grip - rest) / iqr0,
                 'p80_gt_0': u / (len(r80) * len(r0)),
                 'rho': stats.spearmanr(pk.target_pct, pk.peak).statistic})

tab = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tab.round(2).to_csv(OUT, index=False)
print(tab.round(2).to_string(index=False))
lo, hi = tab.loc[tab.range_px.idxmin()], tab.loc[tab.range_px.idxmax()]
print(f'\nmin range {lo.range_px:.2f} px ({lo.participant}), max range {hi.range_px:.2f} px ({hi.participant})')
steps = tab[['rest_px', 'med5_px', 'med20_px', 'med50_px', 'grip80_px']].diff(axis=1).iloc[:, 1:]
print(f'medians rise at every step: {int((steps > 0).all(axis=1).sum())} / {len(tab)} participants')
print(f'range / iqr0: {tab.range_iqr0.min():.2f} - {tab.range_iqr0.max():.2f}, '
      f'p80_gt_0: {tab.p80_gt_0.min():.3f} - {tab.p80_gt_0.max():.3f}, rho: {tab.rho.min():.3f} - {tab.rho.max():.3f}')
