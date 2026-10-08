# notebooks/significance_analysis.py
#
# Is the RMSE gap between Ridge / XGBoost / MLP real, or fold-to-fold noise?
# 10-participant cohort. Reads the six fold_metrics_*.json files only -- no refits, runs in seconds.
#
# Unit of analysis = the PARTICIPANT (n = 10), on both rungs.
#   rung 1: the 98 cross-mount folds are clustered inside 10 participants, so they are not
#           independent. Average each participant's folds first, then test on n = 10.
#           The 98-fold test is printed only as a labelled sensitivity check.
#   rung 3: one fold per participant already, n = 10.
#
# Test: Friedman omnibus (3 models, exact permutation p at n = 10), then paired Wilcoxon signed-rank for each pair,
# Holm-corrected across the 3 pairs. Effect size = Hodges-Lehmann estimate of the paired
# difference with its exact 95% CI -- built from the same signed ranks as Wilcoxon, so the CI
# excludes 0 exactly when the (raw) Wilcoxon test rejects. No resampling, no seed.
# Exact Wilcoxon p-value floor at n = 10 is 2/2^10 = 0.00195.
#
# The 8-participant analysis (journal state, n = 79 folds, Nemenyi) lives in git history,
# commit 350a2fd, and in docs/design/models/model-comparison-significance-test.md.
#
# Run from notebooks/:  python significance_analysis.py

import glob
import itertools
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

MODELS = ['Ridge', 'XGBoost', 'MLP']
FILES = {'Ridge': 'ridge', 'XGBoost': 'xgboost', 'MLP': 'mlp'}
PAIRS = [('Ridge', 'XGBoost'), ('Ridge', 'MLP'), ('XGBoost', 'MLP')]
ALPHA = 0.05


def load_rmse(suffix, key):
    # one column per model, rows matched on key
    merged = None
    for m in MODELS:
        with open(f'fold_metrics_{FILES[m]}{suffix}.json') as f:
            records = json.load(f)
        df = pd.DataFrame([{**{k: r[k] for k in key}, m: r['metrics']['overall']['rmse']} for r in records])
        merged = df if merged is None else merged.merge(df, on=key, how='inner')
    return merged


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = running_max
    return adjusted


def exact_friedman_p(chi2, n, k=3):
    # Exact Friedman p-value: under H0 each participant's ranks are one of the k! orders,
    # all equally likely. Build the full distribution of the rank-sum vector (6^10 = 60M
    # outcomes at n = 10, folded into a few hundred rank-sum states) and count how often
    # chi2 is at least as large as the observed one. The chi2 approximation is unreliable
    # at n = 10, k = 3, so this p is the one the doc reports.
    orders = list(itertools.permutations(range(1, k + 1)))
    dist = {(0,) * k: 1}
    for _ in range(n):
        new = defaultdict(int)
        for sums, count in dist.items():
            for order in orders:
                new[tuple(s + r for s, r in zip(sums, order))] += count
        dist = new
    hits = 0
    for sums, count in dist.items():
        stat = 12 / (n * k * (k + 1)) * sum(s * s for s in sums) - 3 * n * (k + 1)
        if stat >= chi2 - 1e-9:
            hits += count
    return hits / len(orders) ** n


def hodges_lehmann(diff, alpha=ALPHA):
    # HL estimate = median of the n(n+1)/2 Walsh averages (each pair of differences, averaged,
    # including each difference with itself). The CI cuts c Walsh averages off each end, where c
    # comes from the exact null distribution of the Wilcoxon signed-rank statistic.
    n = len(diff)
    walsh = np.sort([(diff[i] + diff[j]) / 2 for i in range(n) for j in range(i, n)])
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1
    for r in range(1, n + 1):          # each rank r is either positive or negative
        counts[r:] = counts[r:] + counts[:-r].copy()
    cdf = np.cumsum(counts) / 2 ** n
    c = int(np.searchsorted(cdf, alpha / 2, side='right')) - 1   # largest k with P(T <= k) <= alpha/2
    return np.median(walsh), walsh[c], walsh[len(walsh) - 1 - c]


def report(df, label):
    X = df[MODELS].to_numpy()
    n = len(X)
    print(f'\n=== {label}  (n = {n}) ===')

    ranks = np.apply_along_axis(stats.rankdata, 1, X)
    wins = np.bincount(X.argmin(axis=1), minlength=3)
    print(f"{'model':8s} {'mean':>7s} {'SD':>6s} {'median':>7s} {'avg rank':>9s} {'wins':>5s}")
    for j, m in enumerate(MODELS):
        print(f'{m:8s} {X[:, j].mean():7.3f} {X[:, j].std(ddof=1):6.3f} {np.median(X[:, j]):7.3f} '
              f'{ranks[:, j].mean():9.2f} {wins[j]:5d}')

    chi2, p_friedman = stats.friedmanchisquare(*X.T)
    if n <= 12:
        print(f'Friedman chi2(2) = {chi2:.3f}, exact p = {exact_friedman_p(chi2, n):.4g}  '
              f'(chi2 approximation p = {p_friedman:.4g})')
    else:
        print(f'Friedman chi2(2) = {chi2:.3f}, p = {p_friedman:.4g}  (chi2 approximation, n large)')

    raw_p = []
    rows = []
    for a, b in PAIRS:
        diff = (df[a] - df[b]).to_numpy()          # negative = a has lower RMSE
        w = stats.wilcoxon(df[a], df[b])
        hl, lo, hi = hodges_lehmann(diff)
        raw_p.append(w.pvalue)
        rows.append((a, b, hl, lo, hi,
                     w.statistic, w.pvalue, int((diff < 0).sum())))

    print(f"{'pair':18s} {'HL diff':>9s} {'HL 95% CI':>18s} {'W':>6s} {'p raw':>8s} {'p Holm':>8s}  a better in")
    for (a, b, md, lo, hi, W, p, k), p_adj in zip(rows, holm(raw_p)):
        flag = '*' if p_adj < 0.05 else ' '
        print(f'{a + " - " + b:18s} {md:+9.3f} [{lo:+7.3f}, {hi:+7.3f}] {W:6.1f} {p:8.4f} {p_adj:8.4f}{flag} {k}/{n}')


# ---- RUNG 1: cross-mount -----------------------------------------------------------
rung1 = load_rmse('', ['participant_id', 'test_session'])
assert len(rung1) == 98, f'expected 98 matched rung-1 folds, got {len(rung1)}'
assert rung1['participant_id'].nunique() == 10

rung1_pp = rung1.groupby('participant_id')[MODELS].mean().reset_index()

report(rung1_pp, 'RUNG 1 -- per-participant mean RMSE  [MAIN RESULT]')
report(rung1, 'RUNG 1 -- 98 folds as independent  [sensitivity only: folds are clustered]')

# 8-participant subset -- rung-1 folds stay inside one participant, so these must equal
# the journal numbers (Ridge 7.06, XGBoost 6.13, MLP 6.44) exactly.
sub8 = rung1[rung1['participant_id'] <= 8]
print('\nRUNG 1 p001-p008 fold means (journal check):',
      ', '.join(f'{m} {sub8[m].mean():.2f}' for m in MODELS))
report(rung1_pp[rung1_pp['participant_id'] <= 8], 'RUNG 1 -- per-participant, p001-p008 only')


# ---- RUNG 3: cross-subject ---------------------------------------------------------
rung3 = load_rmse('_rung3', ['participant_id'])
assert len(rung3) == 10, f'expected 10 matched rung-3 folds, got {len(rung3)}'

report(rung3, 'RUNG 3 -- leave-one-participant-out  [MAIN RESULT]')


# ---- per-participant tables (for the docs) -----------------------------------------
print('\nRUNG 1 per-participant mean RMSE:')
print(rung1_pp.round(2).to_string(index=False))
print('\nRUNG 3 RMSE per held-out participant:')
print(rung3.round(2).to_string(index=False))


# ==== RUNG 4: one-session recalibration (plan: docs/logs/20261008_124119_rung4_recalibration_plan.md) ====
# Pre-registered. Rung 3 + a straight line (a + b * pred) fitted on one enrolment session k, scored
# on the participant's other sessions. A "turn" = one enrolment session k with its test sessions.
# Everything is a PARTICIPANT-level pooled RMSE (n = 10): sqrt(sum(n_j * rmse_j^2) / sum(n_j)).
# Rung-4 and rung-3 numbers come from fold_metrics_rung4.json, rung-1 numbers from the rung-1 files.

print('\n\n' + '#' * 90)
print('# RUNG 4 -- one-session recalibration')
print('#' * 90)

with open('fold_metrics_rung4.json') as f:
    rung4_records = json.load(f)
assert len(rung4_records) == 294, f'expected 294 rung-4 records, got {len(rung4_records)}'


def pooled(n, rmse):
    # pooled RMSE over a set of sessions, weighted by each session's row count
    n = np.asarray(n, dtype=float)
    rmse = np.asarray(rmse, dtype=float)
    return np.sqrt((n * rmse ** 2).sum() / n.sum())


# rung-1 lookup: (model, participant, test session) -> (n, rmse)
rung1_lookup = {}
for m in MODELS:
    with open(f'fold_metrics_{FILES[m]}.json') as f:
        for r in json.load(f):
            rung1_lookup[(m, r['participant_id'], r['test_session'])] = (
                r['metrics']['overall']['n'], r['metrics']['overall']['rmse'])

# rung-4 turn lookup: (model, participant, enrolment session k) -> record
turn = {(r['model'], r['participant_id'], r['enrol_session']): r for r in rung4_records}
PIDS = sorted({r['participant_id'] for r in rung4_records})
assert len(PIDS) == 10
assert all(('Ridge', p, 1) in turn for p in PIDS), 'session 1 must exist for every participant'


def turn_rmse(m, p, k, sessions=None):
    # pooled rung-4, rung-3 and rung-1 RMSE for one turn (optionally only the listed test sessions)
    # rung 1 is scored on the SAME sessions, and the row counts must match
    tests = [t for t in turn[(m, p, k)]['test'] if sessions is None or t['session'] in sessions]
    n = [t['n'] for t in tests]
    n1 = [rung1_lookup[(m, p, t['session'])][0] for t in tests]
    assert n == n1, f'row counts differ between rung 4 and rung 1: {m} p{p} k={k}'
    return (pooled(n, [t['rmse_rung4'] for t in tests]),
            pooled(n, [t['rmse_rung3'] for t in tests]),
            pooled(n, [rung1_lookup[(m, p, t['session'])][1] for t in tests]))


# ---- 3.2 PRIMARY: turn k = 1, sessions 2..K -----------------------------------------
prim = {m: pd.DataFrame([dict(zip(['participant_id', 'rung4', 'rung3', 'rung1'],
                                  (p,) + turn_rmse(m, p, 1))) for p in PIDS]) for m in MODELS}

print('\n=== RUNG 4 PRIMARY -- enrol on session 1, pooled RMSE over sessions 2..K, per participant ===')
table = pd.DataFrame({'participant_id': PIDS})
for m in MODELS:
    for rung in ['rung1', 'rung3', 'rung4']:
        table[f'{m} {rung}'] = prim[m][rung].to_numpy()
print(table.round(2).to_string(index=False))

print('\n=== RUNG 4 PRIMARY family -- exact Wilcoxon, n = 10, Holm over 6 tests ===')
prim_rows = []
for m in MODELS:
    for other in ['rung3', 'rung1']:
        x4, xo = prim[m]['rung4'].to_numpy(), prim[m][other].to_numpy()
        diff = x4 - xo                              # negative = rung 4 has lower RMSE
        w = stats.wilcoxon(x4, xo)
        hl, lo, hi = hodges_lehmann(diff)
        prim_rows.append((f'{m} rung4 - {other}', np.median(diff), hl, lo, hi,
                          w.statistic, w.pvalue, int((diff < 0).sum())))
prim_holm = holm([r[6] for r in prim_rows])
print(f"{'test':22s} {'median diff':>11s} {'HL diff':>8s} {'HL 95% CI':>18s} {'W':>5s} {'p raw':>8s} {'p Holm':>8s}  rung 4 lower")
for (name, md, hl, lo, hi, W, p, k), p_adj in zip(prim_rows, prim_holm):
    flag = '*' if p_adj < ALPHA else ' '
    print(f'{name:22s} {md:+11.3f} {hl:+8.3f} [{lo:+7.3f}, {hi:+7.3f}] {W:5.1f} {p:8.4f} {p_adj:8.4f}{flag} {k}/10')

# ---- 3.3 across models on rung 4 (primary values) -----------------------------------
rung4_models = pd.DataFrame({'participant_id': PIDS, **{m: prim[m]['rung4'].to_numpy() for m in MODELS}})
report(rung4_models, 'RUNG 4 -- across models, primary turn (k = 1)')

# ---- 3.4 asymmetry (sensitivity, pre-specified; separate Holm family over 3 models) --
# forward = enrolment session k before test session j (k < j), backward = k after j (k > j).
# Pool every forward (k, j) session entry of a participant, and every backward entry, over ALL turns.
print('\n=== RUNG 4 ASYMMETRY (sensitivity) -- rung-4 RMSE, forward (k < j) vs backward (k > j), all turns ===')
asym = {}
for m in MODELS:
    fwd, bwd = [], []
    for p in PIDS:
        nf, rf, nb, rb = [], [], [], []
        for k in sorted(k for (mm, pp, k) in turn if mm == m and pp == p):
            for t in turn[(m, p, k)]['test']:
                if k < t['session']:
                    nf.append(t['n']); rf.append(t['rmse_rung4'])
                elif k > t['session']:
                    nb.append(t['n']); rb.append(t['rmse_rung4'])
        fwd.append(pooled(nf, rf))
        bwd.append(pooled(nb, rb))
    asym[m] = (np.array(fwd), np.array(bwd))

asym_rows = []
for m in MODELS:
    fwd, bwd = asym[m]
    diff = fwd - bwd                                # negative = forward has lower RMSE
    w = stats.wilcoxon(fwd, bwd)
    hl, lo, hi = hodges_lehmann(diff)
    asym_rows.append((m, fwd.mean(), bwd.mean(), hl, lo, hi, w.statistic, w.pvalue, int((diff < 0).sum())))
asym_holm = holm([r[7] for r in asym_rows])
print(f"{'model':8s} {'fwd mean':>9s} {'bwd mean':>9s} {'HL diff':>8s} {'HL 95% CI':>18s} {'W':>5s} {'p raw':>8s} {'p Holm':>8s}  fwd lower")
for (m, fm, bm, hl, lo, hi, W, p, k), p_adj in zip(asym_rows, asym_holm):
    verdict = 'order effect detected' if p_adj < ALPHA else 'order effect not detected'
    print(f'{m:8s} {fm:9.3f} {bm:9.3f} {hl:+8.3f} [{lo:+7.3f}, {hi:+7.3f}] {W:5.1f} {p:8.4f} {p_adj:8.4f}  {k}/10  -> {verdict}')

# ---- 3.5 descriptive only, no tests -------------------------------------------------
# a) rotation mean: per-turn pooled rung-4 RMSE, then mean (and min / max) over the turns
print('\n=== RUNG 4 DESCRIPTIVE (a) -- rotation mean over turns, next to the primary value (no test) ===')
rot = pd.DataFrame({'participant_id': PIDS})
for m in MODELS:
    per_turn = {p: [turn_rmse(m, p, k)[0] for k in sorted(k for (mm, pp, k) in turn if mm == m and pp == p)]
                for p in PIDS}
    rot[f'{m} primary'] = prim[m]['rung4'].to_numpy()
    rot[f'{m} rot mean'] = [np.mean(per_turn[p]) for p in PIDS]
    rot[f'{m} min'] = [np.min(per_turn[p]) for p in PIDS]
    rot[f'{m} max'] = [np.max(per_turn[p]) for p in PIDS]
print(rot.round(2).to_string(index=False))

# b) EchoForce-style row: primary turn (k = 1), score ONLY the participant's last session
print('\n=== RUNG 4 DESCRIPTIVE (b) -- EchoForce-style: k = 1, last session only (no test) ===')
last = {}
for m in MODELS:
    rows = []
    for p in PIDS:
        last_session = max(t['session'] for t in turn[(m, p, 1)]['test'])
        rows.append((last_session,) + turn_rmse(m, p, 1, sessions={last_session}))
    last[m] = rows
print(f"{'participant':>11s} {'last s':>6s}  " + '  '.join(f'{m + " r1":>11s} {m + " r3":>11s} {m + " r4":>11s}' for m in MODELS))
for i, p in enumerate(PIDS):
    cells = []
    for m in MODELS:
        s, r4, r3, r1 = last[m][i]
        cells.append(f'{r1:11.2f} {r3:11.2f} {r4:11.2f}')
    print(f'{p:11d} {last["Ridge"][i][0]:6d}  ' + '  '.join(cells))
print('mean +/- population SD over 10 participants (%MVC):')
for m in MODELS:
    for j, rung in [(3, 'rung 1'), (2, 'rung 3'), (1, 'rung 4')]:
        v = np.array([r[j] for r in last[m]])
        print(f'  {m:8s} {rung}: {v.mean():.3f} +/- {np.std(v):.3f}')

# c) kg table: pooled %MVC RMSE (primary turn) * mvc_kg / 100, mean +/- population SD over participants
mvc_kg = {}
for p in PIDS:
    with open(glob.glob(f'../recordings/participant{p:03d}_*/participant.json')[0]) as f:
        mvc_kg[p] = json.load(f)['mvc_kg']
print('\n=== RUNG 4 DESCRIPTIVE (c) -- kg table, primary turn (k = 1), mean +/- population SD, n = 10 ===')
print(f"{'model':8s} {'rung 1 (kg)':>16s} {'rung 3 (kg)':>16s} {'rung 4 (kg)':>16s}")
for m in MODELS:
    cells = []
    for rung in ['rung1', 'rung3', 'rung4']:
        kg = prim[m][rung].to_numpy() * np.array([mvc_kg[p] for p in PIDS]) / 100
        cells.append(f'{kg.mean():.3f} +/- {np.std(kg):.3f}')
    print(f'{m:8s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s}')
print('Table 5.13 rung 1 averages 98 session folds; here every rung is a participant mean over sessions 2..K.')
