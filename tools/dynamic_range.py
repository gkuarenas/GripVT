"""Reproduce thesis Table 5.8, the signal dynamic range per participant.

For every session in notebooks/split_manifest.json:
  frame peak  = largest marker magnitude in a frame (markers.csv)
  trial peak  = largest frame peak inside one trial_id
Per participant:
  rest  = median trial peak over target_pct == 0 trials
  grip  = median trial peak over target_pct == 80 trials
  range = grip - rest, ratio = grip / rest   (all in px)

Usage (from the repo root):
    python tools/dynamic_range.py
Writes docs/results/dynamic_range_table58.csv.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

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
    rows.append({'participant': pid, 'rest_px': rest, 'grip80_px': grip, 'range_px': grip - rest,
                 'ratio': grip / rest, 'n0': len(r0), 'n80': len(r80)})

tab = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tab.round(2).to_csv(OUT, index=False)
print(tab.round(2).to_string(index=False))
lo, hi = tab.loc[tab.range_px.idxmin()], tab.loc[tab.range_px.idxmax()]
print(f'\nmin range {lo.range_px:.2f} px ({lo.participant}), max range {hi.range_px:.2f} px ({hi.participant})')
