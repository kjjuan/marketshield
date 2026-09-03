"""
train.py — Trains the MarketShield forward-volatility model (v3).

Change from v2: the target is no longer the 8 hand-drawn binary stress
windows. It is a continuous regression target — the S&P's realized
volatility over the next 20 trading days (see labels.py). We fit a small
bake-off of models, score them out-of-sample, and keep the winner.

Run order:
    python fetch_data.py     # -> raw_market_data.csv  (needs internet)
    python train.py          # builds features + target, bake-off, saves winner

Outputs:
    model.pkl           — the winning fitted estimator (joblib)
    model.json          — same model in XGBoost's format, IF the winner is xgboost
    model_meta.json     — winner name, target spec, 0-100 calibration, feature list
    feature_stats.json  — feature means/stds, for the simulator
    training_report.txt — full bake-off table + walk-forward CV
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from features import FEATURE_COLUMNS, build_features
from labels import TARGET_HORIZON, TARGET_NAME, add_labels, add_regression_target
from model_utils import RiskScorer, build_candidates, contributions, is_xgb

TEST_FRAC = 0.20
N_WF_FOLDS = 5


def load_dataset(raw_csv="raw_market_data.csv"):
    raw = pd.read_csv(raw_csv, index_col=0, parse_dates=True)
    feats = build_features(raw)
    df = add_regression_target(feats, raw)
    df = add_labels(df)              # 'stress' kept only as an eval yard-stick
    return df, feats


def _metrics(y_true, y_pred, stress):
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rho = float(spearmanr(y_pred, y_true).statistic)
    out = {"rmse": rmse, "mae": mae, "spearman": rho}
    if stress is not None and pd.Series(stress).nunique() > 1:
        out["auc_vs_labels"] = float(roc_auc_score(stress, y_pred))
        out["pr_vs_labels"] = float(average_precision_score(stress, y_pred))
    else:
        out["auc_vs_labels"] = float("nan")
        out["pr_vs_labels"] = float("nan")
    return out


def walk_forward(estimator, X, y, stress, n_folds=N_WF_FOLDS):
    n = len(X)
    edges = np.linspace(int(n * 0.4), n, n_folds + 1, dtype=int)
    rows = []
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        if y.iloc[a:b].isna().all() or b - a < 20:
            continue
        m = clone(estimator)
        m.fit(X.iloc[:a], y.iloc[:a])
        p = m.predict(X.iloc[a:b])
        rows.append(_metrics(y.iloc[a:b], p, stress.iloc[a:b]))
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    return agg, rows


def train_model(df: pd.DataFrame):
    # rows without a full forward window can't be trained/tested on
    train_test = df.dropna(subset=["target"]).copy()
    X = train_test[FEATURE_COLUMNS]
    y = train_test["target"]
    stress = train_test["stress"]

    split_idx = int(len(X) * (1 - TEST_FRAC))
    boundary = X.index[split_idx]
    Xtr, Xte = X.iloc[:split_idx], X.iloc[split_idx:]
    ytr, yte = y.iloc[:split_idx], y.iloc[split_idx:]
    str_te = stress.iloc[split_idx:]

    candidates = build_candidates()
    lines = []
    lines.append(f"target            : {TARGET_NAME} "
                 f"(S&P realized vol, next {TARGET_HORIZON} trading days, annualized %)")
    lines.append(f"rows (with target): {len(X)}   "
                 f"{X.index.min().date()} -> {X.index.max().date()}")
    lines.append(f"train/test split  : {boundary.date()}  "
                 f"({len(Xtr)} / {len(Xte)})")
    lines.append(f"target mean/std   : train {ytr.mean():.1f} / {ytr.std():.1f}   "
                 f"test {yte.mean():.1f} / {yte.std():.1f}")
    lines.append("")
    lines.append("HOLD-OUT (last 20%)          RMSE    MAE  Spearman  AUC-lbl  PR-lbl")

    results = {}
    for name, est in candidates.items():
        est.fit(Xtr, ytr)
        m = _metrics(yte, est.predict(Xte), str_te)
        wf, _ = walk_forward(est, X, y, stress)
        results[name] = {"holdout": m, "walk_forward": wf}
        lines.append(
            f"  {name:24s}{m['rmse']:7.2f}{m['mae']:7.2f}"
            f"{m['spearman']:9.3f}{m['auc_vs_labels']:9.3f}{m['pr_vs_labels']:8.3f}"
        )

    lines.append("")
    lines.append("WALK-FORWARD (5 folds)      Spearman  AUC-lbl  PR-lbl    RMSE")
    for name in candidates:
        wf = results[name]["walk_forward"]
        lines.append(
            f"  {name:24s}{wf['spearman']:9.3f}{wf['auc_vs_labels']:9.3f}"
            f"{wf['pr_vs_labels']:8.3f}{wf['rmse']:8.2f}"
        )

    # winner: best walk-forward separation of the independent crisis labels,
    # tie-broken by walk-forward RMSE
    winner = max(
        candidates,
        key=lambda n: (results[n]["walk_forward"]["auc_vs_labels"],
                       -results[n]["walk_forward"]["rmse"]),
    )
    best = candidates[winner]  # already fitted on train
    scorer = RiskScorer(best, FEATURE_COLUMNS, ytr)

    lines.append("")
    lines.append(f"WINNER: {winner}")
    lines.append(f"risk-score calibration: predicted vol {scorer.lo:.1f} -> 0, "
                 f"{scorer.hi:.1f} -> 100 (linear clip; train P25/P95)")

    report = "\n".join(lines)
    with open("training_report.txt", "w") as f:
        f.write(report + "\n")
    print(report)

    return scorer, winner, Xtr, Xte


def save_artifacts(scorer: RiskScorer, winner: str, X_train: pd.DataFrame):
    joblib.dump(scorer.model, "model.pkl")
    if is_xgb(scorer.model):
        scorer.model.save_model("model.json")
    elif os.path.exists("model.json"):
        os.remove("model.json")  # stale artifact from a previous xgboost winner

    meta = {
        "kind": "regression",
        "target": TARGET_NAME,
        "horizon": TARGET_HORIZON,
        "winner": winner,
        "feature_columns": FEATURE_COLUMNS,
        "trained_through": str(X_train.index.max().date()),
        "risk_calibration": scorer.calib(),
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    feature_stats = {
        col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())}
        for col in FEATURE_COLUMNS
    }
    with open("feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)


def explain_latest(scorer: RiskScorer, X: pd.DataFrame, background: pd.DataFrame,
                   row_index=-1):
    row = X.iloc[row_index]
    return {
        "risk_score": round(scorer.score_one(row), 1),
        "predicted_fwd_vol": round(float(scorer.predict_target(row.to_frame().T)[0]), 1),
        "contributions": {
            k: round(v, 4) for k, v in contributions(scorer.model, row, background).items()
        },
    }


if __name__ == "__main__":
    df, feats = load_dataset()
    scorer, winner, X_train, X_test = train_model(df)
    save_artifacts(scorer, winner, X_train)

    result = explain_latest(scorer, X_test, X_train)
    print("\nMost recent test-period day's risk breakdown:")
    print(json.dumps(result, indent=2))
