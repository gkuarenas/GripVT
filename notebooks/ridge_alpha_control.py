# notebooks/ridge_alpha_control.py
#
# M5: Ridge alpha control for the rung-4 v2 refit (plan: docs/logs/20261012_181201_m5_m6_seed_and_alpha_plan.md, section 5).
#
# sklearn Ridge does not normalize sample_weight, so the refit (sum w = 2 * n_pool) runs at about alpha / 1.98
# relative to the w = 1 control (sum w = N). This script refits the Ridge control with
# alpha_scaled = alpha * N / sum(w_refit) (factor 0.505 to 0.509), so both arms have the same effective penalty.
# For each of the 10 participants:
#   1. refit the control at the stored alpha and check it against fold_metrics_rung4_v2.json (4 decimals)
#   2. refit the control at alpha_scaled and score sessions 2..K
# Output: fold_metrics_rung4_v2_m5.json. The test is the M5 block in significance_analysis.py.
# Run from the notebooks/ folder: python ridge_alpha_control.py   (about 5 min, CPU)

import time
from _common import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

OUT_PATH = CURRENT_FOLDER / 'fold_metrics_rung4_v2_m5.json'
ENROL_SESSION = 1

def make_ridge(alpha):
    return Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])

def pooled_rmse(entries):
    return float(np.sqrt(sum(e['n'] * e['rmse'] ** 2 for e in entries) / sum(e['n'] for e in entries)))

with open(CURRENT_FOLDER / 'fold_metrics_ridge_rung3.json') as f:
    fold_records = {r['participant_id']: r for r in json.load(f)}
with open(CURRENT_FOLDER / 'fold_metrics_rung4_v2.json') as f:
    v2 = {(r['participant_id'], r['arm']): r for r in json.load(f) if r['model'] == 'Ridge'}

SESSIONS = {int(p.removeprefix('participant')): sorted(v['sessions']) for p, v in MANIFEST['participants'].items()}
records = []
for fold in SUBJECT_FOLDS:
    test_p = int(fold['test_participant'].removeprefix('participant'))
    alpha = fold_records[test_p]['best_params']['alpha']
    test_sessions = [s for s in SESSIONS[test_p] if s != ENROL_SESSION]
    train_nums = [int(p.removeprefix('participant')) for p in fold['train_participants']]
    pool_mask = FEATURES['participant_id'].isin(train_nums)
    held_mask = FEATURES['participant_id'] == test_p
    enrol_mask = held_mask & (FEATURES['session_id'] == ENROL_SESSION)
    test_mask = held_mask & (FEATURES['session_id'] != ENROL_SESSION)
    train_mask = pool_mask | enrol_mask
    assert test_p not in train_nums and not (train_mask & test_mask).any()
    is_enrol = enrol_mask[train_mask].to_numpy()
    X_train = FEATURES.loc[train_mask, FEATURE_COLS].to_numpy()
    y_train = FEATURES.loc[train_mask, 'pct_mvc_corr'].to_numpy()
    X_test = FEATURES.loc[test_mask, FEATURE_COLS].to_numpy()
    y_test = FEATURES.loc[test_mask, 'pct_mvc_corr'].to_numpy()
    sess_test = FEATURES.loc[test_mask, 'session_id'].to_numpy()
    n_enrol, n_pool = int(is_enrol.sum()), int((~is_enrol).sum())
    assert (n_enrol, n_pool) == (v2[(test_p, 'refit')]['n_enrol'], v2[(test_p, 'refit')]['n_pool'])
    N = n_pool + n_enrol
    sum_w_refit = n_pool + (n_pool / n_enrol) * n_enrol      # = 2 * n_pool
    factor = N / sum_w_refit
    w1 = np.ones(len(y_train))

    out = {}
    for arm, a in [('control_repro', alpha), ('control_scaled', alpha * factor)]:
        t0 = time.time()
        m = make_ridge(a)
        m.fit(X_train, y_train, scaler__sample_weight=w1, ridge__sample_weight=w1)
        pred = m.predict(X_test)
        entries = []
        for s in test_sessions:
            k = sess_test == s
            err = pred[k] - y_test[k]
            entries.append({'session': int(s), 'n': int(k.sum()),
                            'rmse': float(np.sqrt(np.mean(err ** 2))), 'mae': float(np.mean(np.abs(err)))})
        out[arm] = (a, entries, time.time() - t0)

    want = pooled_rmse(v2[(test_p, 'control')]['test'])
    got = pooled_rmse(out['control_repro'][1])
    if round(got, 4) != round(want, 4):
        raise RuntimeError(f'p{test_p}: control repro {got:.6f} != stored {want:.6f}')
    a, entries, fit_s = out['control_scaled']
    records.append({'model': 'Ridge', 'participant_id': test_p, 'arm': 'control_scaled', 'enrol_session': ENROL_SESSION,
                    'w': 1.0, 'alpha': float(alpha), 'alpha_scaled': float(a), 'factor': float(factor),
                    'n_pool': n_pool, 'n_enrol': n_enrol, 'fit_s': round(fit_s, 1), 'test': entries})
    print(f"p{test_p:02d} alpha={alpha:.4g} factor={factor:.4f} repro ok  control={want:.3f}  "
          f"scaled={pooled_rmse(entries):.3f}  refit={pooled_rmse(v2[(test_p, 'refit')]['test']):.3f}", flush=True)

with open(OUT_PATH, 'w') as f:
    json.dump(records, f, indent=2)
print('done', len(records), 'records ->', OUT_PATH.name)
