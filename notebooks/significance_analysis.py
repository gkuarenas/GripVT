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


# ==== RUNG 4 v2: weighted refit (plan: docs/logs/20261008_140743_rung4_v2_plan.md) ====
# Pre-registered. Rung-3 model refit on the 9 training participants + session 1 of the held-out
# participant, session 1 weighted w = N_pool / N_enrol. Control = same refit with w = 1.
# Scored on sessions 2..K. Participant-level pooled RMSE (n = 10).
# Primary family: refit vs rung 3 (Holm over 3). Secondary family: refit vs rung 1 and refit vs
# control (Holm over 6). Gap closed = HL(rung3 - refit) / HL(rung3 - rung1), descriptive only.
# v1 (straight line) appears only as a descriptive row for the appendix, never as a comparator.

print('\n\n' + '#' * 90)
print('# RUNG 4 v2 -- weighted refit')
print('#' * 90)

with open('fold_metrics_rung4_v2.json') as f:
    v2_records = json.load(f)
assert len(v2_records) == 60, f'expected 60 rung-4 v2 records, got {len(v2_records)}'
v2 = {(r['model'], r['participant_id'], r['arm']): r for r in v2_records}


def v2_rmse(m, p):
    # pooled refit, control, rung-3, rung-1 and v1 RMSE over sessions 2..K, on identical frames
    refit = v2[(m, p, 'refit')]['test']
    control = v2[(m, p, 'control')]['test']
    v1_tests = {t['session']: t for t in turn[(m, p, 1)]['test']}
    sessions = [t['session'] for t in refit]
    assert [t['session'] for t in control] == sessions == sorted(v1_tests)
    n = [t['n'] for t in refit]
    assert n == [t['n'] for t in control] == [v1_tests[s]['n'] for s in sessions] \
        == [rung1_lookup[(m, p, s)][0] for s in sessions], f'row counts differ: {m} p{p}'
    return {
        'refit': pooled(n, [t['rmse'] for t in refit]),
        'control': pooled(n, [t['rmse'] for t in control]),
        'rung3': pooled(n, [v1_tests[s]['rmse_rung3'] for s in sessions]),
        'rung1': pooled(n, [rung1_lookup[(m, p, s)][1] for s in sessions]),
        'v1': pooled(n, [v1_tests[s]['rmse_rung4'] for s in sessions]),
    }


v2_prim = {m: pd.DataFrame([{'participant_id': p, **v2_rmse(m, p)} for p in PIDS]) for m in MODELS}
for m in MODELS:   # rung 1 and rung 3 here must equal the v1 primary values exactly
    assert np.allclose(v2_prim[m]['rung3'], prim[m]['rung3']) and np.allclose(v2_prim[m]['rung1'], prim[m]['rung1'])

print('\n=== RUNG 4 v2 -- enrol on session 1, pooled RMSE over sessions 2..K, per participant ===')
for m in MODELS:
    t = v2_prim[m][['participant_id', 'rung1', 'rung3', 'control', 'refit']].copy()
    t['w'] = [v2[(m, p, 'refit')]['w'] for p in PIDS]
    print(m)
    print(t.round(2).to_string(index=False))
    print('mean', t[['rung1', 'rung3', 'control', 'refit']].mean().round(3).to_dict())


def v2_family(tests, label):
    rows = []
    for m, a, b in tests:
        xa, xb = v2_prim[m][a].to_numpy(), v2_prim[m][b].to_numpy()
        diff = xa - xb                                  # negative = a has lower RMSE
        w = stats.wilcoxon(xa, xb)
        hl, lo, hi = hodges_lehmann(diff)
        rows.append((f'{m} {a} - {b}', np.median(diff), hl, lo, hi, w.statistic, w.pvalue, int((diff < 0).sum())))
    p_holm = holm([r[6] for r in rows])
    print(f'\n=== {label} ===')
    print(f"{'test':26s} {'median diff':>11s} {'HL diff':>8s} {'HL 95% CI':>18s} {'W':>5s} {'p raw':>8s} {'p Holm':>8s}  a lower")
    for (name, md, hl, lo, hi, W, p, k), p_adj in zip(rows, p_holm):
        flag = '*' if p_adj < ALPHA else ' '
        print(f'{name:26s} {md:+11.3f} {hl:+8.3f} [{lo:+7.3f}, {hi:+7.3f}] {W:5.1f} {p:8.4f} {p_adj:8.4f}{flag} {k}/10')
    return rows, p_holm


# ---- 7.1 PRIMARY: refit vs rung 3, Holm over 3 -----------------------------------------
v2_primary, v2_primary_holm = v2_family([(m, 'refit', 'rung3') for m in MODELS],
                                        'RUNG 4 v2 PRIMARY family -- refit vs rung 3, exact Wilcoxon, n = 10, Holm over 3')

# ---- 7.2 SECONDARY: refit vs rung 1 and refit vs control, Holm over 6 -------------------
v2_family([(m, 'refit', other) for m in MODELS for other in ['rung1', 'control']],
          'RUNG 4 v2 SECONDARY family -- refit vs rung 1 and vs control, exact Wilcoxon, n = 10, Holm over 6')
print('A non-significant refit - rung1 result means "difference not detected", never "equal".')

# ---- 7.3 descriptive only, no tests ----------------------------------------------------
# a) gap closed = HL(rung3 - refit) / HL(rung3 - rung1)
print('\n=== RUNG 4 v2 DESCRIPTIVE (a) -- share of the rung 3 -> rung 1 gap closed (HL estimates, no test) ===')
for m in MODELS:
    gain = hodges_lehmann((v2_prim[m]['rung3'] - v2_prim[m]['refit']).to_numpy())[0]
    gap = hodges_lehmann((v2_prim[m]['rung3'] - v2_prim[m]['rung1']).to_numpy())[0]
    print(f'{m:8s} HL(r3 - refit) = {gain:+.3f}   HL(r3 - r1) = {gap:+.3f}   gap closed = {gain / gap:.1%}')

# b) kg table: rungs 1, 3 and the refit, mean +/- population SD over participants
print('\n=== RUNG 4 v2 DESCRIPTIVE (b) -- kg table, sessions 2..K, mean +/- population SD, n = 10 ===')
print(f"{'model':8s} {'rung 1 (kg)':>16s} {'rung 3 (kg)':>16s} {'refit (kg)':>16s}")
for m in MODELS:
    cells = []
    for rung in ['rung1', 'rung3', 'refit']:
        kg = v2_prim[m][rung].to_numpy() * np.array([mvc_kg[p] for p in PIDS]) / 100
        cells.append(f'{kg.mean():.3f} +/- {np.std(kg):.3f}')
    print(f'{m:8s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s}')

# c) v1 row for the appendix: the straight-line recalibration on the same sessions
print('\n=== RUNG 4 v2 DESCRIPTIVE (c) -- v1 (linear recalibration) on the same sessions, appendix only ===')
for m in MODELS:
    print(f"{m:8s} mean %MVC RMSE  rung3 {v2_prim[m]['rung3'].mean():.3f}  v1 {v2_prim[m]['v1'].mean():.3f}  "
          f"refit {v2_prim[m]['refit'].mean():.3f}")

# ---- 7.4 rotation trigger (pre-registered) ---------------------------------------------
hits = [m for (name, md, hl, lo, hi, W, p, k), p_adj, m in zip(v2_primary, v2_primary_holm, MODELS)
        if p_adj < ALPHA and hl < 0]
print(f'\nRotation trigger (primary Holm p < {ALPHA} with refit lower, in >= 1 model): '
      f'{"TRIGGERED by " + ", ".join(hits) if hits else "NOT triggered"}')


# ==== M3: forward-only subject-dependent ceiling (plan: docs/logs/20261012_145403_forward_only_rung1_plan.md) ====
# Forward rung 1 = test session j trained on the same participant's sessions before j only (j = 2..K).
# Family A: forward vs current rung 1.  Family B: rung 4 v2 (weighted refit) vs forward.
# All scores are participant-level pooled RMSE over sessions 2..K, identical frames (n checked). n = 10.
# Exact Wilcoxon, Holm within each family over 3 models. Skipped if the forward file is not complete.
# Path override for testing: environment variable FORWARD_SD_PATH.

import os

print('\n\n' + '#' * 90)
print('# M3 -- forward-only subject-dependent ceiling')
print('#' * 90)

FORWARD_PATH = os.environ.get('FORWARD_SD_PATH', 'fold_metrics_forward_sd.json')
forward_records = []
if os.path.exists(FORWARD_PATH):
    with open(FORWARD_PATH) as f:
        forward_records = json.load(f)

if len(forward_records) < 264:
    print(f'M3 block skipped: {FORWARD_PATH} has {len(forward_records)} records '
          f'(missing file counts as 0), 264 needed (3 models x 88 folds).')
else:
    assert len(forward_records) == 264, f'expected 264 forward records, got {len(forward_records)}'
    # forward lookup: (model, participant, test session) -> (n, rmse)
    fwd_lookup = {}
    for r in forward_records:
        key = (r['model'], r['participant_id'], r['test_session'])
        assert key not in fwd_lookup, f'duplicate forward record {key}'
        fwd_lookup[key] = (r['metrics']['overall']['n'], r['metrics']['overall']['rmse'])

    m3 = {}
    for m in MODELS:
        rows = []
        for p in PIDS:
            sessions = sorted(s for (mm, pp, s) in fwd_lookup if mm == m and pp == p)
            assert sessions, f'no forward folds: {m} p{p}'
            n = [fwd_lookup[(m, p, s)][0] for s in sessions]
            # same frames as rung 4 v2 (sessions 2..K) and as the current rung 1
            assert sessions == [t['session'] for t in v2[(m, p, 'refit')]['test']], f'sessions differ from v2: {m} p{p}'
            assert n == [t['n'] for t in v2[(m, p, 'refit')]['test']], f'frame counts differ from v2: {m} p{p}'
            assert n == [rung1_lookup[(m, p, s)][0] for s in sessions], f'frame counts differ from rung 1: {m} p{p}'
            rows.append({'participant_id': p,
                         'forward': pooled(n, [fwd_lookup[(m, p, s)][1] for s in sessions]),
                         'rung1': pooled(n, [rung1_lookup[(m, p, s)][1] for s in sessions]),
                         'refit': v2_prim[m]['refit'].iloc[PIDS.index(p)],
                         'rung3': v2_prim[m]['rung3'].iloc[PIDS.index(p)]})
        m3[m] = pd.DataFrame(rows)
        assert np.allclose(m3[m]['rung1'], v2_prim[m]['rung1']), f'rung 1 pooled differs from v2 table: {m}'

    print('\n=== M3 -- pooled RMSE over sessions 2..K, per participant (%MVC) ===')
    table = pd.DataFrame({'participant_id': PIDS})
    for m in MODELS:
        for col in ['forward', 'rung1', 'refit']:
            table[f'{m} {col}'] = m3[m][col].to_numpy()
    print(table.round(2).to_string(index=False))
    print('mean', {f'{m} {c}': round(m3[m][c].mean(), 3) for m in MODELS for c in ['forward', 'rung1', 'refit']})

    def m3_family(tests, label):
        rows = []
        for m, a, b in tests:
            xa, xb = m3[m][a].to_numpy(), m3[m][b].to_numpy()
            diff = xa - xb                                  # negative = a has lower RMSE
            w = stats.wilcoxon(xa, xb, method='exact')
            hl, lo, hi = hodges_lehmann(diff)
            rows.append((f'{m} {a} - {b}', np.median(diff), hl, lo, hi, w.statistic, w.pvalue, int((diff < 0).sum())))
        p_holm = holm([r[6] for r in rows])
        print(f'\n=== {label} ===')
        print(f"{'test':26s} {'median diff':>11s} {'HL diff':>8s} {'HL 95% CI':>18s} {'W':>5s} {'p raw':>8s} {'p Holm':>8s}  a lower")
        for (name, md, hl, lo, hi, W, p, k), p_adj in zip(rows, p_holm):
            flag = '*' if p_adj < ALPHA else ' '
            verdict = 'difference detected' if p_adj < ALPHA else 'difference not detected (n = 10)'
            print(f'{name:26s} {md:+11.3f} {hl:+8.3f} [{lo:+7.3f}, {hi:+7.3f}] {W:5.1f} {p:8.4f} {p_adj:8.4f}{flag} {k}/10  {verdict}')
        return rows, p_holm

    m3_family([(m, 'forward', 'rung1') for m in MODELS],
              'M3 FAMILY A -- forward vs current rung 1, exact Wilcoxon, n = 10, Holm over 3')
    m3_family([(m, 'refit', 'forward') for m in MODELS],
              'M3 FAMILY B -- rung 4 v2 (refit) vs forward, exact Wilcoxon, n = 10, Holm over 3')

    # descriptive only, no tests: gap closed, same HL definition as RUNG 4 v2 DESCRIPTIVE (a)
    print('\n=== M3 DESCRIPTIVE -- share of the rung 3 -> ceiling gap closed by the refit (HL estimates, no test) ===')
    for m in MODELS:
        gain = hodges_lehmann((m3[m]['rung3'] - m3[m]['refit']).to_numpy())[0]
        for ceiling in ['rung1', 'forward']:
            gap = hodges_lehmann((m3[m]['rung3'] - m3[m][ceiling]).to_numpy())[0]
            print(f'{m:8s} ceiling = {ceiling:8s} HL(r3 - refit) = {gain:+.3f}   HL(r3 - ceiling) = {gap:+.3f}   '
                  f'gap closed = {gain / gap:.1%}')


# ==== M6: seed variance (plan: docs/logs/20261012_181201_m5_m6_seed_and_alpha_plan.md, section 4) ====
# Pre-registered. MLP (rung 1, rung 3, refit, control) and XGBoost (rung 1) retrained with seeds 0..19 and
# the stored best_params (05e_seed_variance.ipynb). Per seed, build the participant value exactly as above;
# then average it over the seeds. Ridge is deterministic and XGBoost rung 3 / refit / control were not
# reseeded (plan section 2, item 5): they keep their seed-42 values. Same tests and families as above.
# Keep/revert rule (plan section 2, item 7), on the rung-1 test only: keep the MLP unless XGBoost - MLP
# has Holm p < 0.05 with XGBoost lower. Skipped if the seed files are not complete.

print('\n\n' + '#' * 90)
print('# M6 -- seed variance (seed-mean values)')
print('#' * 90)

SEED_FILES = ['fold_metrics_mlp_seeds.json', 'fold_metrics_xgboost_seeds.json']
seed_records = []
for path in SEED_FILES:
    if os.path.exists(path):
        with open(path) as f:
            seed_records += json.load(f)
SEEDS = sorted({r['seed'] for r in seed_records})
seed_counts = {s: sum(r['seed'] == s for r in seed_records) for s in SEEDS}

if len(SEEDS) < 2 or any(c != 226 for c in seed_counts.values()):
    print(f'M6 block skipped: need >= 2 complete seeds (226 records each), found {seed_counts}.')
else:
    assert 42 not in SEEDS
    seed_lookup = {}                                     # (model, arm, seed, participant, test session) -> record
    for r in seed_records:
        key = (r['model'], r['arm'], r['seed'], r['participant_id'], r.get('test_session'))
        assert key not in seed_lookup, f'duplicate seed record {key}'
        seed_lookup[key] = r
    sessions_of = {p: sorted(s for (m, a, sd, pp, s) in seed_lookup if m == 'MLP' and a == 'rung1' and sd == SEEDS[0] and pp == p)
                   for p in PIDS}

    def seed_value(m, arm, seed, p):
        # one participant value for one seed, same definition as the seed-42 tables above
        if arm == 'rung1':                               # participant mean of the fold RMSE (headline)
            return np.mean([seed_lookup[(m, 'rung1', seed, p, s)]['metrics']['overall']['rmse'] for s in sessions_of[p]])
        if arm == 'rung1_2K':                            # pooled over sessions 2..K, frames = v2
            tests = v2[(m, p, 'refit')]['test']
            recs = [seed_lookup[(m, 'rung1', seed, p, t['session'])]['metrics']['overall'] for t in tests]
            assert [r['n'] for r in recs] == [t['n'] for t in tests], f'frame counts differ: {m} p{p} seed {seed}'
            return pooled([r['n'] for r in recs], [r['rmse'] for r in recs])
        if arm == 'rung3':                               # all sessions of the held-out participant
            return seed_lookup[(m, 'rung3', seed, p, None)]['metrics']['overall']['rmse']
        if arm == 'rung3_2K':                            # rung 3 scored on sessions 2..K
            tests = [t for t in seed_lookup[(m, 'rung3', seed, p, None)]['test'] if t['session'] != 1]
        else:                                            # refit / control, the 05c layout
            tests = seed_lookup[(m, arm, seed, p, None)]['test']
        assert [t['n'] for t in tests] == [t['n'] for t in v2[(m, p, 'refit')]['test']], f'frame counts differ: {m} {arm} p{p}'
        return pooled([t['n'] for t in tests], [t['rmse'] for t in tests])

    SEED_ARMS = [('MLP', 'rung1'), ('XGBoost', 'rung1'), ('MLP', 'rung1_2K'), ('MLP', 'rung3'),
                 ('MLP', 'rung3_2K'), ('MLP', 'refit'), ('MLP', 'control')]
    seed_tab = {(m, arm): pd.DataFrame({s: [seed_value(m, arm, s, p) for p in PIDS] for s in SEEDS}, index=PIDS)
                for m, arm in SEED_ARMS}             # rows = participants, columns = seeds
    seed_mean = {k: t.mean(axis=1) for k, t in seed_tab.items()}

    # seed-42 values with the same definitions, for the side-by-side table
    s42 = {('MLP', 'rung1'): rung1_pp.set_index('participant_id')['MLP'],
           ('XGBoost', 'rung1'): rung1_pp.set_index('participant_id')['XGBoost'],
           ('MLP', 'rung1_2K'): v2_prim['MLP'].set_index('participant_id')['rung1'],
           ('MLP', 'rung3'): rung3.set_index('participant_id')['MLP'],
           ('MLP', 'rung3_2K'): v2_prim['MLP'].set_index('participant_id')['rung3'],
           ('MLP', 'refit'): v2_prim['MLP'].set_index('participant_id')['refit'],
           ('MLP', 'control'): v2_prim['MLP'].set_index('participant_id')['control']}

    print(f'\n=== M6 seed summary -- {len(SEEDS)} seeds ({SEEDS[0]}..{SEEDS[-1]}), mean over 10 participants (%MVC) ===')
    print(f"{'arm':18s} {'seed-42 mean':>12s} {'seed mean':>10s} {'mean seed SD':>13s} {'max seed SD':>12s}  max at")
    for k, t in seed_tab.items():
        sd = t.std(axis=1, ddof=1)
        print(f'{k[0] + " " + k[1]:18s} {s42[k].mean():12.3f} {seed_mean[k].mean():10.3f} {sd.mean():13.3f} '
              f'{sd.max():12.3f}  p{sd.idxmax():03d}')

    print('\n=== M6 seed SD per participant (%MVC) ===')
    sd_table = pd.DataFrame({f'{m} {a}': seed_tab[(m, a)].std(axis=1, ddof=1) for m, a in SEED_ARMS})
    sd_table.index = [f'p{p:03d}' for p in PIDS]
    print(sd_table.round(3).to_string())

    # ---- 4.1 rung 1, seed-mean MLP and XGBoost, Ridge deterministic ----------------------
    m6_rung1 = rung1_pp.copy()
    for m in ['MLP', 'XGBoost']:
        m6_rung1[m] = seed_mean[(m, 'rung1')].reindex(m6_rung1['participant_id']).to_numpy()
    report(m6_rung1, 'M6 RUNG 1 -- per-participant mean RMSE, MLP and XGBoost = seed mean  [KEEP/REVERT TEST]')

    raw_p = [stats.wilcoxon(m6_rung1[a], m6_rung1[b]).pvalue for a, b in PAIRS]
    p_xm = holm(raw_p)[PAIRS.index(('XGBoost', 'MLP'))]
    hl_xm = hodges_lehmann((m6_rung1['XGBoost'] - m6_rung1['MLP']).to_numpy())[0]
    revert = p_xm < ALPHA and hl_xm < 0
    print(f'\nKeep/revert rule: XGBoost - MLP HL = {hl_xm:+.3f}, Holm p = {p_xm:.4f} -> '
          f'{"REVERT to XGBoost (XGBoost significantly better)" if revert else "KEEP the MLP (XGBoost not significantly better)"}')

    # ---- 4.2 rung 3, seed-mean MLP, Ridge and XGBoost at seed 42 --------------------------
    m6_rung3 = rung3.copy()
    m6_rung3['MLP'] = seed_mean[('MLP', 'rung3')].reindex(m6_rung3['participant_id']).to_numpy()
    report(m6_rung3, 'M6 RUNG 3 -- MLP = seed mean, Ridge and XGBoost = seed 42')

    # ---- 4.3 rung 4 v2 families, MLP columns = seed mean ----------------------------------
    m6_v2 = {m: v2_prim[m].copy() for m in MODELS}
    for col, arm in [('rung1', 'rung1_2K'), ('rung3', 'rung3_2K'), ('refit', 'refit'), ('control', 'control')]:
        m6_v2['MLP'][col] = seed_mean[('MLP', arm)].reindex(m6_v2['MLP']['participant_id']).to_numpy()

    def m6_family(tests, label):
        rows = []
        for m, a, b in tests:
            xa, xb = m6_v2[m][a].to_numpy(), m6_v2[m][b].to_numpy()
            diff = xa - xb                                  # negative = a has lower RMSE
            w = stats.wilcoxon(xa, xb)
            hl, lo, hi = hodges_lehmann(diff)
            rows.append((f'{m} {a} - {b}', np.median(diff), hl, lo, hi, w.statistic, w.pvalue, int((diff < 0).sum())))
        p_holm = holm([r[6] for r in rows])
        print(f'\n=== {label} ===')
        print(f"{'test':26s} {'median diff':>11s} {'HL diff':>8s} {'HL 95% CI':>18s} {'W':>5s} {'p raw':>8s} {'p Holm':>8s}  a lower")
        for (name, md, hl, lo, hi, W, p, k), p_adj in zip(rows, p_holm):
            flag = '*' if p_adj < ALPHA else ' '
            verdict = 'difference detected' if p_adj < ALPHA else 'difference not detected (n = 10)'
            print(f'{name:26s} {md:+11.3f} {hl:+8.3f} [{lo:+7.3f}, {hi:+7.3f}] {W:5.1f} {p:8.4f} {p_adj:8.4f}{flag} {k}/10  {verdict}')
        return rows, p_holm

    print('\nMLP sessions 2..K, mean over participants (seed mean):',
          {c: round(m6_v2['MLP'][c].mean(), 3) for c in ['rung1', 'rung3', 'control', 'refit']})
    m6_family([(m, 'refit', 'rung3') for m in MODELS],
              'M6 RUNG 4 v2 PRIMARY -- refit vs rung 3, MLP = seed mean, exact Wilcoxon, n = 10, Holm over 3')
    m6_family([(m, 'refit', other) for m in MODELS for other in ['rung1', 'control']],
              'M6 RUNG 4 v2 SECONDARY -- refit vs rung 1 and vs control, MLP = seed mean, Holm over 6')


# ==== M5: Ridge alpha control (plan: docs/logs/20261012_181201_m5_m6_seed_and_alpha_plan.md, section 5) ====
# Pre-registered. The Ridge refit runs at about alpha / 1.98 relative to the w = 1 control, because sklearn
# does not normalize sample_weight. ridge_alpha_control.py refits the control at alpha * N / sum(w_refit).
# Test: Ridge refit vs scaled control, exact Wilcoxon, n = 10, its own family (Holm over 1).
# Scaled vs original control is descriptive only. Skipped if the M5 file is missing.

print('\n\n' + '#' * 90)
print('# M5 -- Ridge alpha control')
print('#' * 90)

if not os.path.exists('fold_metrics_rung4_v2_m5.json'):
    print('M5 block skipped: fold_metrics_rung4_v2_m5.json not found (run ridge_alpha_control.py).')
else:
    with open('fold_metrics_rung4_v2_m5.json') as f:
        m5 = {r['participant_id']: r for r in json.load(f)}
    assert sorted(m5) == list(PIDS)
    m5_t = v2_prim['Ridge'][['participant_id', 'refit', 'control']].copy()
    m5_t['control_scaled'] = [pooled([t['n'] for t in m5[p]['test']], [t['rmse'] for t in m5[p]['test']])
                              for p in m5_t['participant_id']]
    m5_t['factor'] = [m5[p]['factor'] for p in m5_t['participant_id']]
    print(m5_t.round(3).to_string(index=False))
    print('mean', m5_t[['refit', 'control', 'control_scaled']].mean().round(3).to_dict())
    print(f"\n{'test':32s} {'HL diff':>8s} {'HL 95% CI':>18s} {'p exact':>8s}  a lower")
    for a, b, note in [('refit', 'control_scaled', 'M5 test, Holm over 1'),
                       ('control_scaled', 'control', 'descriptive')]:
        xa, xb = m5_t[a].to_numpy(), m5_t[b].to_numpy()
        diff = xa - xb
        w = stats.wilcoxon(xa, xb, method='exact')
        hl, lo, hi = hodges_lehmann(diff)
        verdict = ('difference detected' if w.pvalue < ALPHA else 'difference not detected (n = 10)') + f'  [{note}]'
        print(f'{a + " - " + b:32s} {hl:+8.3f} [{lo:+7.3f}, {hi:+7.3f}] {w.pvalue:8.4f}  {int((diff < 0).sum())}/10  {verdict}')
