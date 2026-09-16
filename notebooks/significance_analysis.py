import json
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

def load(f):
    with open(f) as fh:
        return json.load(fh)

ridge1 = load('fold_metrics_ridge.json')
xgb1 = load('fold_metrics_xgboost.json')
mlp1 = load('fold_metrics_mlp.json')

ridge3 = load('fold_metrics_ridge_rung3.json')
xgb3 = load('fold_metrics_xgboost_rung3.json')
mlp3 = load('fold_metrics_mlp_rung3.json')

def to_df(d, model, key_fields):
    rows = []
    for r in d:
        row = {kf: r[kf] for kf in key_fields}
        row['model'] = model
        row['rmse'] = r['metrics']['overall']['rmse']
        row['mae'] = r['metrics']['overall']['mae']
        row['r2'] = r['metrics']['overall']['r2']
        rows.append(row)
    return pd.DataFrame(rows)

# ---- RUNG1: cross-mount, key = participant_id + test_session ----
key1 = ['participant_id', 'test_session']
df_r1 = to_df(ridge1, 'Ridge', key1)
df_x1 = to_df(xgb1, 'XGBoost', key1)
df_m1 = to_df(mlp1, 'MLP', key1)

merged1 = df_r1.merge(df_x1, on=key1, suffixes=('_ridge', '_xgb'))
merged1 = merged1.merge(df_m1, on=key1)
merged1 = merged1.rename(columns={'rmse': 'rmse_mlp', 'mae': 'mae_mlp', 'r2': 'r2_mlp', 'model': 'model_mlp'})

print("=== RUNG1 fold alignment check ===")
print("Ridge folds:", len(df_r1), "XGB folds:", len(df_x1), "MLP folds:", len(df_m1))
print("Merged (inner join) rows:", len(merged1))
assert len(merged1) == 79, "Fold alignment FAILED for rung1 -- expected 79 matched rows"
print("Fold alignment OK: 79/79 folds matched 1:1 across all three model files.\n")

rmse_ridge = merged1['rmse_ridge'].values
rmse_xgb = merged1['rmse_xgb'].values
rmse_mlp = merged1['rmse_mlp'].values

# Sanity check against manuscript-published RMSE means
print("Sanity check (rung1 RMSE means): Ridge=%.4f XGBoost=%.4f MLP=%.4f" % (
    rmse_ridge.mean(), rmse_xgb.mean(), rmse_mlp.mean()))
print("(manuscript: Ridge 7.06, XGBoost 6.13, MLP 6.44)\n")

# ---- Friedman test on rung1 RMSE ----
stat, p = stats.friedmanchisquare(rmse_ridge, rmse_xgb, rmse_mlp)
k = 3
N = len(rmse_ridge)
dof = k - 1
print("=== Friedman test (rung1, RMSE, N=%d folds, k=%d models) ===" % (N, k))
print("chi-square statistic = %.6f, dof = %d, p-value = %.10g" % (stat, dof, p))

alpha = 0.05
if p < alpha:
    print("\nResult is significant at alpha=0.05. Running Nemenyi post-hoc test.\n")
    # scikit-posthocs expects a matrix (folds x models) block design
    data_matrix = np.column_stack([rmse_ridge, rmse_xgb, rmse_mlp])
    nemenyi = sp.posthoc_nemenyi_friedman(data_matrix)
    nemenyi.columns = ['Ridge', 'XGBoost', 'MLP']
    nemenyi.index = ['Ridge', 'XGBoost', 'MLP']
    print("Nemenyi post-hoc adjusted p-values:")
    print(nemenyi)

    # average ranks (for reporting) -- rank 1 = best (lowest RMSE) per fold
    ranks = np.apply_along_axis(stats.rankdata, 1, data_matrix)
    avg_ranks = ranks.mean(axis=0)
    print("\nAverage ranks (1=best/lowest RMSE): Ridge=%.4f XGBoost=%.4f MLP=%.4f" % tuple(avg_ranks))

    # Critical difference per Demsar 2006, for reporting alongside
    # q_alpha for k=3 at alpha=0.05 (two-tailed Nemenyi) = 2.343
    q_alpha = 2.343
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    print("Critical Difference (Demsar 2006, alpha=0.05, k=3, N=%d): CD = %.4f" % (N, CD))
    print("Rank differences: Ridge-XGBoost=%.4f, Ridge-MLP=%.4f, XGBoost-MLP=%.4f" % (
        avg_ranks[0]-avg_ranks[1], avg_ranks[0]-avg_ranks[2], avg_ranks[1]-avg_ranks[2]))
else:
    print("\nResult is NOT significant at alpha=0.05. No post-hoc test performed.")

# ---- RUNG3: descriptive only, n=8, underpowered ----
print("\n=== RUNG3 (cross-participant, n=8 folds) -- descriptive only, no formal test ===")
key3 = ['participant_id', 'fold']
df_r3 = to_df(ridge3, 'Ridge', key3)
df_x3 = to_df(xgb3, 'XGBoost', key3)
df_m3 = to_df(mlp3, 'MLP', key3)
merged3 = df_r3.merge(df_x3, on=key3, suffixes=('_ridge', '_xgb')).merge(df_m3, on=key3)
merged3 = merged3.rename(columns={'rmse': 'rmse_mlp', 'mae': 'mae_mlp', 'r2': 'r2_mlp'})
print("Merged rung3 rows:", len(merged3), "(expect 8)")
assert len(merged3) == 8

for label, col_prefix in [('rmse', 'rmse'), ('mae', 'mae'), ('r2', 'r2')]:
    for model, suf in [('Ridge', '_ridge'), ('XGBoost', '_xgb'), ('MLP', '_mlp')]:
        v = merged3[col_prefix + suf].values
        print("%s %s: mean=%.4f sd=%.4f range=[%.4f, %.4f]" % (
            model, label, v.mean(), v.std(ddof=1), v.min(), v.max()))
