"""Reproduce the MVC and fatigue numbers in the thesis Section 5.5.7 limitations.

Post-hoc descriptive checks for the Discussion only. No model input changes.

For every session in notebooks/split_manifest.json:
  trial peak = largest pct_mvc_corr inside one trial_id (features_table.csv, the models' label)
Per participant:
  med0..med80 = median trial peak %MVC at each target level (achieved force)
  rho80, p80  = Spearman correlation of session number against 80% trial peak (fatigue proxy)
  first80/last80 = median 80% trial peak in the first and last half of the sessions
  n_over100   = trials whose peak exceeded 100 %MVC
  visit_min   = first to last frame of the visit, minutes
From mvc.txt (Biopac export of the three MVC contractions):
  mvc_peaks, mvc_cv = the three contraction peaks and their coefficient of variation (%)
  mvc_gaps_s        = seconds between the contraction peaks
Cohort: Wilcoxon signed-rank test of first80 against last80, n = 10.

Usage (from the repo root):
    python tools/mvc_checks.py
Writes docs/results/mvc_checks.csv.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'results', 'mvc_checks.csv')
MVC_DT = 0.005                                        # Biopac sample period, s

with open(os.path.join(ROOT, 'notebooks', 'split_manifest.json')) as f:
    manifest = json.load(f)['participants']


def mvc_contractions(path):
    """Return the three largest contraction peaks (kg) and their sample indices, in time order."""
    lines = open(path).read().splitlines()
    start = [k for k, line in enumerate(lines) if line.startswith('0,')][0]
    v = np.array([float(line.split(',')[1]) for line in lines[start:] if line.count(',') >= 1 and line.split(',')[1]])
    pk, _ = find_peaks(v, distance=int(10 / MVC_DT), prominence=0.4 * v.max())
    top = sorted(sorted(pk, key=lambda k: -v[k])[:3])
    return v[top], np.array(top)


rows = []
for pid in sorted(manifest):
    num = int(pid.replace('participant', ''))
    pdir = glob.glob(os.path.join(ROOT, 'recordings', f'participant{num:03d}_*'))
    assert len(pdir) == 1, f'expected one folder for p{num}, got {pdir}'
    pdir = pdir[0]
    with open(os.path.join(pdir, 'participant.json')) as f:
        mvc_kg = json.load(f)['mvc_kg']

    peaks, t0, t1 = [], [], []
    for s in manifest[pid]['sessions']:
        folder = glob.glob(os.path.join(pdir, f'session{s}_*'))
        assert len(folder) == 1, f'expected one folder for p{num} s{s}, got {folder}'
        ft = pd.read_csv(os.path.join(folder[0], 'features_table.csv'),
                         usecols=['timestamp_ms', 'trial_id', 'target_pct', 'pct_mvc_corr'])
        t0.append(ft.timestamp_ms.min())
        t1.append(ft.timestamp_ms.max())
        ft = ft.dropna(subset=['trial_id'])
        pk = ft.groupby('trial_id').agg(target=('target_pct', 'first'), peak=('pct_mvc_corr', 'max'))
        pk['session'] = s
        peaks.append(pk)
    pk = pd.concat(peaks)
    med = pk.groupby('target')['peak'].median()
    p80 = pk[pk.target == 80]
    rho, p = stats.spearmanr(p80.session, p80.peak)
    half = p80.session.median()

    mvc_peaks, idx = mvc_contractions(os.path.join(pdir, 'mvc.txt'))
    assert abs(mvc_peaks.mean() - mvc_kg) < 0.01, f'{pid}: mvc.txt peaks do not reproduce mvc_kg'
    rows.append({'participant': pid, 'mvc_kg': mvc_kg,
                 **{f'med{int(k)}': v for k, v in med.items()},
                 'rho80': rho, 'p80': p,
                 'first80': p80[p80.session <= half].peak.median(),
                 'last80': p80[p80.session > half].peak.median(),
                 'n_over100': int((pk.peak > 100).sum()), 'max_peak': pk.peak.max(),
                 'visit_min': (max(t1) - min(t0)) / 60000, 'n_trials': len(pk),
                 'mvc_best_of_3': mvc_peaks.max(),
                 'mvc_cv': 100 * mvc_peaks.std(ddof=1) / mvc_peaks.mean(),
                 'mvc_gaps_s': ' '.join(f'{g:.0f}' for g in np.diff(idx) * MVC_DT)})

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
pd.set_option('display.width', 250)
print(df.round(2).to_string(index=False))
print('\ncohort median of participant medians:',
      df[[c for c in df if c.startswith('med')]].median().round(1).to_dict())
print(f'cohort mean MVC {df.mvc_kg.mean():.2f} kg, best of three {df.mvc_best_of_3.mean():.2f} kg')
print(f'MVC CV range {df.mvc_cv.min():.1f} to {df.mvc_cv.max():.1f} %')
print('80% peak falling, p < 0.01:', list(df.participant[(df.rho80 < 0) & (df.p80 < 0.01)]))
print('first vs last half:', stats.wilcoxon(df.first80, df.last80))
print('wrote', OUT)
