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
# Holm-corrected across the 3 pairs. Paired bootstrap 95% CI on the mean difference.
# Exact Wilcoxon p-value floor at n = 10 is 2/2^10 = 0.00195.
#
# The 8-participant analysis (journal state, n = 79 folds, Nemenyi) lives in git history,
# commit 350a2fd, and in docs/design/models/model-comparison-significance-test.md.
#
# Run from notebooks/:  python significance_analysis.py

import itertools
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

MODELS = ['Ridge', 'XGBoost', 'MLP']
FILES = {'Ridge': 'ridge', 'XGBoost': 'xgboost', 'MLP': 'mlp'}
PAIRS = [('Ridge', 'XGBoost'), ('Ridge', 'MLP'), ('XGBoost', 'MLP')]
N_BOOT = 10000
SEED = 42


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
    rng = np.random.default_rng(SEED)
    for a, b in PAIRS:
        diff = (df[a] - df[b]).to_numpy()          # negative = a has lower RMSE
        w = stats.wilcoxon(df[a], df[b])
        boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        raw_p.append(w.pvalue)
        rows.append((a, b, diff.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5),
                     w.statistic, w.pvalue, int((diff < 0).sum())))

    print(f"{'pair':18s} {'mean diff':>9s} {'95% CI':>18s} {'W':>6s} {'p raw':>8s} {'p Holm':>8s}  a better in")
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
