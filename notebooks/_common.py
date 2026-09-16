# notebooks/_common.py
#
# Shared by 03_ridge.ipynb / 04_xgboost.ipynb / 05_mlp.ipynb / 06_baseline_comparison.ipynb.
# Every notebook does `from _common import *` as its first cell. Whatever this file
# produces must be byte-identical across all four notebooks, or the model comparison
# is void (docs/logs/20260826_224433_plan_notebook03_split.md §3).
#
# Roadmap (build one section at a time, this session covers SECTION 0 only):
#   SECTION 0 -- paths, seed, load split_manifest.json
#   SECTION 1 -- load every features_table.csv, concat, insert participant_id/session_id
#   SECTION 2 -- read folds from the manifest, assert they match what got loaded
#   SECTION 3 -- drop velocity-NaN rows, assert the new row count
#   SECTION 4 -- feature matrix column list + channel_groups index
#   SECTION 5 -- evaluate() -- RMSE / MAE / R2, by-phase, by-level
#   SECTION 6 -- inner_cv_score() -- the no-leak nested-CV helper
#   (MLflow is NOT here -- each notebook owns its own sqlite db, per the split plan)


# ============================================================
# SECTION 0 -- paths, seed, manifest
# ============================================================
# GOAL: give every notebook the same ROOT_FOLDER, the same RANDOM_SEED, and one
# already-parsed copy of split_manifest.json, so nothing downstream re-derives a fold.

import pandas as pd
import numpy as np
import json
from pathlib import Path

CURRENT_FOLDER = Path.cwd()
ROOT_FOLDER = CURRENT_FOLDER.parent
print(CURRENT_FOLDER)
assert (ROOT_FOLDER / 'main.py').exists(), f"repo root is not {ROOT_FOLDER}"

RANDOM_SEED = 42        # Same with split_manifest.json

MANIFEST_PATH = CURRENT_FOLDER / 'split_manifest.json'
with open(MANIFEST_PATH) as f:
    MANIFEST = json.load(f)

assert MANIFEST['random_seed'] == RANDOM_SEED
assert MANIFEST['calibration_mode'] == 'per_wear'
assert MANIFEST['totals']['participants'] == len(MANIFEST['participants'])

print(MANIFEST['totals'])


# ============================================================
# SECTION 1 -- load every features_table.csv, concat, tag rows
# ============================================================
# GOAL: one DataFrame, every session of every participant, each row knowing which
# participant and session it came from. This is the table every model trains on.

recordings_dir = ROOT_FOLDER / 'recordings'

def session_dir(p, s):
    p_folder = f"participant{p:03d}_*"
    s_folder = f"session{s}_*"

    p_hits = list(recordings_dir.glob(p_folder))       # list, not next() -> a 2nd hit is visible
    assert len(p_hits) == 1, (p, p_hits)
    p_dir = p_hits[0]

    s_hits = list(p_dir.glob(s_folder))
    assert len(s_hits) == 1, (p, s, s_hits)

    return s_hits[0]

frames = []
for p, s in MANIFEST['participants'].items():
    p_num = int(p.removeprefix('participant'))
    session_list = s['sessions']

    for s_num in session_list:
        # print(p_num, s_num)
        df = pd.read_csv(session_dir(p_num, s_num) / 'features_table.csv')
        df['participant_id'] = p_num
        df['session_id'] = int(s_num)
        frames.append(df)

FEATURES = pd.concat(frames, ignore_index=True)
assert len(FEATURES) == sum(v['n_rows'] for v in MANIFEST['participants'].values()), f"Expected: {sum(v['n_rows'] for v in MANIFEST['participants'].values())}, got: {len(FEATURES)}"
assert FEATURES.shape[1] == 808, f"Expected 808, got {FEATURES.shape[1]}"
print(len(FEATURES), FEATURES.shape)


# ============================================================
# SECTION 2 -- read folds from the manifest, assert they match what got loaded
# ============================================================
# GOAL: one shared list of session-folds (rung 1 -- the SO4 evaluation every baseline
# notebook trains against) and the manifest's subject-folds passed through untouched
# (rung 3/4 -- notebook 06's job, not 03/04/05's). Then prove both actually describe
# FEATURES, not stale numbers.

SESSION_FOLDS = []
for p, s in MANIFEST['participants'].items():
    p_num = int(p.removeprefix('participant'))
    for fold_dict in s['session_folds']:
        SESSION_FOLDS.append(dict(fold_dict, participant_id=p_num))

SUBJECT_FOLDS = MANIFEST['subject_folds']

assert len(SESSION_FOLDS) == MANIFEST['totals']['session_folds']

for fold in SESSION_FOLDS:
    assert fold['test_session'] not in fold['train_sessions']
    assert fold['test_session'] not in fold['inner_val_sessions']
    assert set(fold['inner_val_sessions']) <= set(fold['train_sessions'])

for p, s in MANIFEST['participants'].items():
    p_num = int(p.removeprefix('participant'))
    manifest_sessions = set(s['sessions'])
    fold_test_sessions = {f['test_session'] for f in SESSION_FOLDS if f['participant_id'] == p_num}
    features_sessions = set(FEATURES.loc[FEATURES['participant_id'] == p_num, 'session_id'].unique())
    assert fold_test_sessions == manifest_sessions, (p_num, fold_test_sessions, manifest_sessions)
    assert fold_test_sessions == features_sessions, (p_num, fold_test_sessions, features_sessions)

print(len(SESSION_FOLDS), len(SUBJECT_FOLDS))


# ============================================================
# SECTION 3 -- drop velocity-NaN rows, assert the new row count
# ============================================================
# GOAL: velocity needs a previous frame to diff against, so the first frame of every
# trial has no velocity yet. Drop exactly that row per trial, and lock the resulting
# row count in as a guard against a future change silently altering the row set.

vel_cols = [c for c in FEATURES.columns if c.startswith('vx_') or c.startswith('vy_')]
assert len(vel_cols) == 198, f"Expected 198 velocity columns, got {len(vel_cols)}"

nan_counts_per_row = FEATURES[vel_cols].isna().sum(axis=1)
assert nan_counts_per_row.isin([0, len(vel_cols)]).all()

rows_before = len(FEATURES)
trial_sizes = FEATURES.groupby(['participant_id', 'session_id', 'trial_id']).size()
n_trials = len(trial_sizes)
# 1-row trials vanish entirely once their only (NaN-velocity) row is dropped --
# verified 2026-08-27, 8 such trials incl. the documented session2/trial13.
# TODO: check the cause of the 7 undocumented ones (participants 5, 6, 8).
n_vanishing_trials = (trial_sizes == 1).sum()

FEATURES = FEATURES.dropna(subset=vel_cols).reset_index(drop=True)

assert rows_before - len(FEATURES) == n_trials
assert FEATURES[vel_cols].isna().sum().sum() == 0
assert FEATURES.groupby(['participant_id', 'session_id', 'trial_id']).ngroups == n_trials - n_vanishing_trials

print(rows_before, len(FEATURES), n_trials, n_vanishing_trials)


# ============================================================
# SECTION 4 -- feature matrix column list + channel_groups index
# ============================================================
# GOAL: one definitive list of the columns a model is actually allowed to see
# (FEATURE_COLS), and a partition of that list into named groups (CHANNEL_GROUPS) --
# the thing notebook 06's ablation loops over. Every model must use the exact same
# FEATURE_COLS or the comparison is void.

prefixes = ('dx_', 'dy_', 'dA_', 'mag_', 'strain_x_', 'strain_y_', 'vx_', 'vy_')
FEATURE_COLS = [c for c in FEATURES.columns if c.startswith(prefixes)] + ['max_disp']
assert len(FEATURE_COLS) == 793

CHANNEL_GROUPS = {
    'dx_dy':    [c for c in FEATURE_COLS if 'dx' in c or 'dy' in c],
    'dA':       [c for c in FEATURE_COLS if 'dA' in c],
    'mag':      [c for c in FEATURE_COLS if 'mag' in c],
    'strain':   [c for c in FEATURE_COLS if 'strain_x' in c or 'strain_y' in c],
    'velocity': [c for c in FEATURE_COLS if 'vx' in c or 'vy' in c],
    'max_disp': ['max_disp']
}

all_grouped = [c for g in CHANNEL_GROUPS.values() for c in g]
assert sorted(all_grouped) == sorted(FEATURE_COLS)   # covers everything
assert len(all_grouped) == len(set(all_grouped))

print({k: len(v) for k, v in CHANNEL_GROUPS.items()})


# ============================================================
# SECTION 5 -- evaluate() -- RMSE / MAE / R2, by-phase, by-level
# ============================================================
# GOAL: one function every model's fold loop calls the same way, so results are
# comparable by construction, not by convention. Headline metric is RMSE in %MVC.
#
# Verified against the real data (not the pilot doc, which assumed 3 phases):
#   phase   -> 2 values: 'pre_cue', 'post_cue' -- PLUS 628 rows with phase = NaN,
#              all belonging to 8 whole trials that have no to_cue data at all.
#   level (target_pct) -> 5 values: 0.0, 5.0, 20.0, 50.0, 80.0 -- PLUS 33 rows with
#              target_pct = NaN, a different (mostly overlapping but not identical)
#              set of rows than the phase-NaN ones.
# Both NaN groups are real rows that must be accounted for, not silently dropped by
# a naive groupby -- a groupby already skips NaN keys, which would make your
# per-slice counts not sum back to the overall count.
#
# Do this, in order:
#
# 1. A small internal helper, used 8+ times below, so write it once. You're on
#    sklearn 1.7.2, which has root_mean_squared_error built in -- use it directly,
#    plus mean_absolute_error and r2_score:
#      from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
#
#      def _metrics(y_true, y_pred):
#          y_true = np.asarray(y_true, dtype=float)
#          y_pred = np.asarray(y_pred, dtype=float)
#          return {
#              'rmse': root_mean_squared_error(y_true, y_pred),
#              'mae': mean_absolute_error(y_true, y_pred),
#              'r2': r2_score(y_true, y_pred),
#              'n': len(y_true),
#          }
#
# 2. def evaluate(y_true, y_pred, phase, level):
#    Convert all four inputs to numpy arrays first (np.asarray) so boolean masking
#    works regardless of whether the caller passed a pandas Series or a plain array.
#
# 3. result['overall'] = _metrics(y_true, y_pred) -- the full set, no filtering.
#
# 4. result['by_phase'] -- a dict keyed by phase value. Loop over
#    sorted(pd.unique(phase[~pd.isna(phase)])) -- i.e. skip NaN explicitly, don't
#    let groupby do it silently -- and for each value, mask and call _metrics.
#    Also store result['n_missing_phase'] = count of NaN phase rows.
#
# 5. result['by_level'] -- same pattern, over level instead of phase.
#    result['n_missing_level'] = count of NaN level rows.
#
# 6. Two accounting asserts, proving no row got silently lost:
#      assert sum(v['n'] for v in result['by_phase'].values()) + result['n_missing_phase'] == result['overall']['n']
#      assert sum(v['n'] for v in result['by_level'].values()) + result['n_missing_level'] == result['overall']['n']
#
# 7. return result
#
# Smoke-test it once you write it: call evaluate(FEATURES['pct_mvc_corr'],
# FEATURES['pct_mvc_corr'], FEATURES['phase'], FEATURES['target_pct']) -- predicting
# the label with itself should give rmse=0, mae=0, r2=1 in every slice. That is the
# cheapest correctness check available before any model exists.

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return{
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'n': len(y_true)
    }


def evaluate(y_true, y_pred, phase, level):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    phase = np.asarray(phase)
    level = np.asarray(level)

    result = {}

    result['overall'] = metrics(y_true, y_pred)

    non_nan_phase = phase[~pd.isna(phase)]
    result['by_phase'] = {}
    for val in sorted(pd.unique(non_nan_phase)):
        mask = phase == val
        result['by_phase'][val] = metrics(y_true[mask], y_pred[mask])
    result['n_missing_phase'] = int(pd.isna(phase).sum())

    non_nan_level = level[~pd.isna(level)]
    result['by_level'] = {}
    for val in sorted(pd.unique(non_nan_level)):
        mask = level == val
        result['by_level'][val] = metrics(y_true[mask], y_pred[mask])
    result['n_missing_level'] = int(pd.isna(level).sum())

    assert sum(v['n'] for v in result['by_phase'].values()) + result['n_missing_phase'] == result['overall']['n']
    assert sum(v['n'] for v in result['by_level'].values()) + result['n_missing_level'] == result['overall']['n']

    return result


_check = evaluate(FEATURES['pct_mvc_corr'], FEATURES['pct_mvc_corr'], FEATURES['phase'], FEATURES['target_pct'])
assert np.isclose(_check['overall']['rmse' if 'rmse' in _check['overall'] else 'RMSE'], 0)
assert np.isclose(_check['overall']['r2' if 'r2' in _check['overall'] else 'R2'], 1)
print('evaluate() self-check passed:', _check['overall'])


# ============================================================
# SECTION 6 -- inner_cv_score() -- the no-leak nested-CV helper
# ============================================================
# GOAL: the one function every model's Optuna objective() calls to score a set of
# hyperparameters, without ever letting the outer test session touch training or
# inner validation. This is the whole point of nesting.
#
# fold is one entry of SESSION_FOLDS: {'fold', 'test_session', 'train_sessions',
# 'inner_val_sessions', 'participant_id'}. inner_val_sessions is the "inner-3
# rotation" -- 3 of the fold's 8 train_sessions, not full LOSO over all 8, chosen to
# keep 79-fold nested tuning affordable.
#
# make_model(params) must return an unfit model with .fit(X, y) / .predict(X) --
# a bare estimator or a Pipeline (e.g. StandardScaler + Ridge), the caller's choice.
# feature_cols defaults to the full FEATURE_COLS but can be overridden -- this is
# what lets notebook 06 reuse this exact function for the channel-group ablation.

from sklearn.linear_model import Ridge

def inner_cv_score(fold, params, make_model, feature_cols=None):
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    p_num = fold['participant_id']
    test_session = fold['test_session']
    train_sessions = fold['train_sessions']

    scores = []
    for val_session in fold['inner_val_sessions']:
        # leak guard 1: structural -- the outer test session can never be the inner
        # validation session or appear in the inner training sessions
        assert val_session != test_session
        inner_train_sessions = [s for s in train_sessions if s != val_session]
        assert test_session not in inner_train_sessions

        train_mask = (FEATURES['participant_id'] == p_num) & (FEATURES['session_id'].isin(inner_train_sessions))
        val_mask = (FEATURES['participant_id'] == p_num) & (FEATURES['session_id'] == val_session)

        # leak guard 2: data-level -- prove it on the actual rows, not just the plan
        assert not (FEATURES.loc[train_mask, 'session_id'] == test_session).any()
        assert not (FEATURES.loc[val_mask, 'session_id'] == test_session).any()

        X_train = FEATURES.loc[train_mask, feature_cols]
        y_train = FEATURES.loc[train_mask, 'pct_mvc_corr']
        X_val = FEATURES.loc[val_mask, feature_cols]
        y_val = FEATURES.loc[val_mask, 'pct_mvc_corr']

        model = make_model(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        scores.append(root_mean_squared_error(y_val, y_pred))

    return float(np.mean(scores))


# Smoke-test with a throwaway Ridge on fold 0 -- proves the function runs end to end
# and the leak guards don't fire on a legitimate fold. Not a real tuning run.
_smoke_score = inner_cv_score(SESSION_FOLDS[0], {'alpha': 1.0}, lambda params: Ridge(**params))
print('inner_cv_score() self-check, fold 0:', _smoke_score)


# ============================================================
# SECTION 2b -- inner_val_participants -- rung 3's inner rotation
# ============================================================
# GOAL: SUBJECT_FOLDS has no inner-validation rotation field (unlike SESSION_FOLDS,
# which reads 'inner_val_sessions' straight from the manifest). Derived here instead
# of in the manifest -- deterministic, fixed ring over the 8 participant numbers, no
# RNG. Decision grilled 2026-08-28. SUBJECT_FOLDS itself is left untouched.

def _ring_next(p, n_total, k=3):
    return [((p - 1 + i) % n_total) + 1 for i in range(1, k + 1)]

assert _ring_next(8, 8) == [1, 2, 3]
assert _ring_next(1, 8) == [2, 3, 4]

N_PARTICIPANTS = MANIFEST['totals']['participants']
assert N_PARTICIPANTS == 8

INNER_VAL_PARTICIPANTS = {}   # fold number -> [participant nums]
for fold in SUBJECT_FOLDS:
    test_p = int(fold['test_participant'].removeprefix('participant'))
    train_ps = [int(p.removeprefix('participant')) for p in fold['train_participants']]

    inner_val = _ring_next(test_p, N_PARTICIPANTS, k=3)

    assert test_p not in inner_val
    assert set(inner_val) <= set(train_ps)
    assert len(inner_val) == 3

    INNER_VAL_PARTICIPANTS[fold['fold']] = inner_val

from collections import Counter
_inner_val_balance = Counter(p for vals in INNER_VAL_PARTICIPANTS.values() for p in vals)
assert all(_inner_val_balance[p] == 3 for p in range(1, N_PARTICIPANTS + 1)), _inner_val_balance

print('fold 0 inner val:', INNER_VAL_PARTICIPANTS[0])
print('fold 7 inner val:', INNER_VAL_PARTICIPANTS[7])
print('balance:', dict(sorted(_inner_val_balance.items())))


# ============================================================
# SECTION 6b -- inner_cv_score_subject() -- rung 3's no-leak nested-CV helper
# ============================================================
# GOAL: the subject-level sibling of inner_cv_score() -- same leak-guard shape, but
# the held-out unit is a whole PARTICIPANT (every session they have), not one session
# of one participant. This is what rung 3's Optuna objective() calls.

def inner_cv_score_subject(fold, inner_val_participants, params, make_model, feature_cols=None):
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    test_participant_num = int(fold['test_participant'].removeprefix('participant'))
    train_participant_nums = [int(p.removeprefix('participant')) for p in fold['train_participants']]

    scores = []
    for val_p in inner_val_participants:
        # leak guard 1: structural -- the outer test participant can never be the
        # inner validation participant or appear in the inner training participants
        assert val_p != test_participant_num
        inner_train_participants = [p for p in train_participant_nums if p != val_p]
        assert test_participant_num not in inner_train_participants

        # no session_id filtering -- rung 3 trains/validates on a participant's FULL
        # data (every session they have). This is the one structural difference from
        # inner_cv_score(), which filters to specific sessions within one participant.
        train_mask = FEATURES['participant_id'].isin(inner_train_participants)
        val_mask = FEATURES['participant_id'] == val_p

        # leak guard 2: data-level -- prove it on the actual rows, not just the plan
        assert not (FEATURES.loc[train_mask, 'participant_id'] == test_participant_num).any()
        assert not (FEATURES.loc[val_mask, 'participant_id'] == test_participant_num).any()

        X_train = FEATURES.loc[train_mask, feature_cols]
        y_train = FEATURES.loc[train_mask, 'pct_mvc_corr']
        X_val = FEATURES.loc[val_mask, feature_cols]
        y_val = FEATURES.loc[val_mask, 'pct_mvc_corr']

        model = make_model(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        scores.append(root_mean_squared_error(y_val, y_pred))

    return float(np.mean(scores))


# Smoke-test with a throwaway Ridge on fold 0 -- proves the function runs end to end,
# the leak guards don't fire on a legitimate fold, and gives a first wall-clock data
# point for rung 3's fold-0 timing check (training set here is ~7-8x rung 1's).
import time
_t0 = time.time()
_smoke_score_subject = inner_cv_score_subject(
    SUBJECT_FOLDS[0], INNER_VAL_PARTICIPANTS[0], {'alpha': 1.0}, lambda params: Ridge(**params)
)
_t1 = time.time()
print(f'inner_cv_score_subject() self-check, fold 0: {_smoke_score_subject} ({_t1 - _t0:.1f}s)')
