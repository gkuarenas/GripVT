# notebooks/inference_timing.py
#
# Single-frame inference latency for Ridge / XGBoost / MLP on the 10 rung-3 folds.
#
# For each model and each rung-3 fold:
#   1. refit the model with that fold's best_params (from fold_metrics_*_rung3.json),
#      on the same training rows the notebook used (GPU for XGBoost/MLP, like 04/05)
#   2. check the refit RMSE against the RMSE in the JSON (proves it is the same model)
#   3. move the fitted model to CPU, 1 thread
#   4. call predict() on ONE row at a time -- the way the live app would call it
#      (WARMUP calls discarded, then N_CALLS timed calls on random test rows)
#
# Timed: model predict only (scaler + model for Ridge/MLP, booster for XGBoost).
# Not timed: feature computation -- all three models share it, so it cannot change the ranking.
#
# Output: timing_rung3.json, checkpointed after every (model, fold) so a crash resumes.
# Run from notebooks/:  python inference_timing.py   (about 30-60 min)

from _common import *

import gc
import time
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
import platform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WARMUP = 200
N_CALLS = 2000
OUT_PATH = CURRENT_FOLDER / 'timing_rung3.json'
TORCH_DEFAULT_THREADS = torch.get_num_threads()


# ============================================================
# Model builders -- copied from 03 / 04 / 05, unchanged
# ============================================================

def make_ridge(params):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=params['alpha'])),
    ])


def make_xgboost(params):
    return XGBRegressor(
        **params,
        random_state=RANDOM_SEED,
        device='cuda',
        tree_method='hist',
        n_jobs=-1,
    )


class MLPTorch(nn.Module):
    def __init__(self, n_features, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPRegressor:
    # same as 05_mlp.ipynb, except predict() uses whatever device model_ is on,
    # so the fitted model can be moved to CPU for timing
    def __init__(self, hidden_dims, dropout, lr, weight_decay, batch_size, n_epochs, random_state=RANDOM_SEED):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.model_ = MLPTorch(X.shape[1], self.hidden_dims, self.dropout).to(DEVICE)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        X_t = torch.from_numpy(X).to(DEVICE)
        y_t = torch.from_numpy(y).to(DEVICE)
        n = len(X_t)

        self.model_.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n, device=DEVICE)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                pred = self.model_(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                opt.step()
        del X_t, y_t
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        device = next(self.model_.parameters()).device
        X_t = torch.from_numpy(X).to(device)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_t)
        return pred.cpu().numpy()


def hidden_dims_from(width, n_layers):
    dims = []
    w = width
    for _ in range(n_layers):
        dims.append(max(w, 8))
        w = max(w // 2, 8)
    return dims


def make_mlp(params):
    hidden_dims = hidden_dims_from(params['width'], params['n_layers'])
    return Pipeline([
        ('scale', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_dims=hidden_dims,
            dropout=params['dropout'],
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            random_state=RANDOM_SEED,
        )),
    ])


MODELS = {
    'Ridge': ('fold_metrics_ridge_rung3.json', make_ridge),
    'XGBoost': ('fold_metrics_xgboost_rung3.json', make_xgboost),
    'MLP': ('fold_metrics_mlp_rung3.json', make_mlp),
}


# ============================================================
# Main loop -- resumable
# ============================================================

if OUT_PATH.exists():
    with open(OUT_PATH) as f:
        out = json.load(f)
else:
    out = {
        'hardware': {'cpu': platform.processor(), 'os': platform.platform(),
                     'torch': torch.__version__, 'gpu_for_fit': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'settings': {'warmup': WARMUP, 'n_calls': N_CALLS, 'threads': 1, 'batch_rows': 1},
        'results': [],
    }

done = {(r['model'], r['participant_id']) for r in out['results']}

for model_name, (json_name, make_model) in MODELS.items():
    with open(CURRENT_FOLDER / json_name) as f:
        fold_records = {r['participant_id']: r for r in json.load(f)}

    for fold in SUBJECT_FOLDS:
        test_p = int(fold['test_participant'].removeprefix('participant'))
        if (model_name, test_p) in done:
            continue
        record = fold_records[test_p]
        params = record['best_params']

        # 1. refit on the same training rows as the notebook
        train_nums = [int(p.removeprefix('participant')) for p in fold['train_participants']]
        train_mask = FEATURES['participant_id'].isin(train_nums)
        test_mask = FEATURES['participant_id'] == test_p
        assert not (FEATURES.loc[train_mask, 'participant_id'] == test_p).any()

        X_train = FEATURES.loc[train_mask, FEATURE_COLS].to_numpy()
        y_train = FEATURES.loc[train_mask, 'pct_mvc_corr'].to_numpy()
        X_test = FEATURES.loc[test_mask, FEATURE_COLS].to_numpy()
        y_test = FEATURES.loc[test_mask, 'pct_mvc_corr'].to_numpy()

        t0 = time.time()
        model = make_model(params)
        model.fit(X_train, y_train)
        fit_s = time.time() - t0
        del X_train, y_train
        gc.collect()

        # 2. same model as the notebook? (GPU training is not bit-exact, so a small gap is expected)
        refit_rmse = float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))
        json_rmse = record['metrics']['overall']['rmse']

        # 3. move to CPU, 1 thread
        if model_name == 'XGBoost':
            model.set_params(device='cpu', n_jobs=1)
        if model_name == 'MLP':
            model.named_steps['mlp'].model_.cpu()
            torch.cuda.empty_cache()
        torch.set_num_threads(1)

        # 4. one row per call
        rng = np.random.default_rng(RANDOM_SEED)
        rows = X_test[rng.integers(0, len(X_test), WARMUP + N_CALLS)]
        latencies_ms = []
        with threadpool_limits(limits=1):
            for i in range(WARMUP + N_CALLS):
                x = rows[i:i + 1]
                t = time.perf_counter()
                model.predict(x)
                dt = (time.perf_counter() - t) * 1000.0
                if i >= WARMUP:
                    latencies_ms.append(dt)
        lat = np.array(latencies_ms)

        result = {
            'model': model_name,
            'participant_id': test_p,
            'best_params': params,
            'fit_seconds_gpu_or_cpu': fit_s,
            'refit_rmse': refit_rmse,
            'json_rmse': json_rmse,
            'latency_ms_median': float(np.median(lat)),
            'latency_ms_p95': float(np.percentile(lat, 95)),
            'latency_ms_mean': float(lat.mean()),
            'latency_ms_min': float(lat.min()),
        }
        out['results'].append(result)
        with open(OUT_PATH, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"{model_name:8s} p{test_p:02d}  rmse refit={refit_rmse:.3f} json={json_rmse:.3f}  "
              f"latency median={result['latency_ms_median']:.3f} ms  p95={result['latency_ms_p95']:.3f} ms  (fit {fit_s:.0f}s)")

        del model, X_test, y_test, rows
        gc.collect()
        torch.set_num_threads(TORCH_DEFAULT_THREADS)   # full threads again for the next fit

print(f"done -- {len(out['results'])} / {3 * len(SUBJECT_FOLDS)} (model, fold) rows in {OUT_PATH.name}")
