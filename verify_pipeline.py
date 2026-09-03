"""
verify_pipeline.py — Verifies the pipeline runs end-to-end using SYNTHETIC data.

This does NOT validate model quality (synthetic data is random) — it only
proves fetch_data -> features -> target -> train -> simulate all execute
without errors. Run this for pipeline/dependency verification; use
fetch_data.py + train.py with real data for the actual model.
"""

import json

import numpy as np
import pandas as pd

np.random.seed(0)
dates = pd.date_range("2005-01-01", "2024-01-01", freq="B")
n = len(dates)

# random-walk-ish synthetic prices, just to exercise the code paths
sp500 = 1000 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, n)))
agg = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.002, n)))
gld = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.008, n)))
eem = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.012, n)))
vix = np.clip(15 + np.cumsum(np.random.normal(0, 0.5, n)), 8, 80)
yc = np.random.normal(1.0, 0.8, n)
credit = np.clip(2 + np.cumsum(np.random.normal(0, 0.03, n)), 0.5, 8)

raw = pd.DataFrame(
    {"VIX": vix, "SP500": sp500, "Bonds": agg, "Gold": gld, "EM": eem,
     "YieldCurveSpread": yc, "CreditSpread": credit},
    index=dates,
)
raw.to_csv("raw_market_data.csv")
print(f"Synthetic raw data written: {raw.shape}")

# --- run the real pipeline modules against this synthetic data ---
from features import FEATURE_COLUMNS, build_features
from labels import add_labels, add_regression_target
from model_utils import RiskScorer
from train import explain_latest, save_artifacts, train_model

feats = build_features(raw)
df = add_labels(add_regression_target(feats, raw))
print(f"Features + target built: {df.shape}, "
      f"target mean = {df['target'].mean():.1f}, stress rate = {df['stress'].mean():.2%}")

scorer, winner, X_train, X_test = train_model(df)
save_artifacts(scorer, winner, X_train)
print(f"\nWinner: {winner}")

result = explain_latest(scorer, X_test, X_train)
print("Sample risk breakdown (synthetic data, not meaningful — pipeline check only):")
print(json.dumps(result, indent=2))

# --- simulator check ---
from simulate import load_scorer, score_scenario

sim_scorer = load_scorer()
current_row = feats.iloc[-1]
sim_result = score_scenario(
    sim_scorer, current_row,
    {"VIX": 25, "CreditSpread": 2.0, "SP500_return_20d": -12, "SP500_drawdown": -18},
    mode="abs", background=feats[FEATURE_COLUMNS].tail(250),
)
print("\nScenario simulator check:")
print(json.dumps(sim_result, indent=2))

print("\n[OK] Pipeline ran end-to-end without errors.")
