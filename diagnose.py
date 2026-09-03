"""
diagnose.py — Is the v3 forward-vol model actually learning, and does it work
as a stress detector out-of-sample?

The target is now continuous (S&P realized vol over the next 20 days), so the
checks are:

  1. Split summary + where the independent crisis labels fall.
  2. Single-feature baselines — Spearman vs. the target, and AUC of each raw
     feature vs. the hand labels. The model has to beat these.
  3. Blocked walk-forward CV — Spearman, RMSE, and AUC-vs-labels per fold, so
     the score isn't riding on one lucky period.
  4. Leave-one-crisis-out — hide a whole stress episode (+60d margin) from
     training, then see how high the model's risk score runs during it.
  5. Test-period risk-score timeline vs. the labels.

Reuses raw_market_data.csv + model_meta.json (run fetch_data.py, then train.py).
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from features import FEATURE_COLUMNS, build_features
from labels import STRESS_WINDOWS, add_labels, add_regression_target
from model_utils import RiskScorer, build_candidates


def load():
    raw = pd.read_csv("raw_market_data.csv", index_col=0, parse_dates=True)
    feats = build_features(raw)
    df = add_labels(add_regression_target(feats, raw))
    meta = json.load(open("model_meta.json"))
    return df, meta


def _winner_estimator(meta):
    name = meta["winner"]
    return name, build_candidates()[name]


# --------------------------------------------------------------------------
def summarise_split(df, meta):
    print("=" * 74)
    print(f"1. SPLIT & LABEL COVERAGE   (winner: {meta['winner']}, target: {meta['target']})")
    print("=" * 74)
    d = df.dropna(subset=["target"])
    X = d[FEATURE_COLUMNS]
    n = len(X)
    split_idx = int(n * 0.8)
    boundary = X.index[split_idx]
    print(f"rows with target     : {n}   {X.index.min().date()} -> {X.index.max().date()}")
    print(f"80/20 split boundary : {boundary.date()}")
    print(f"target (fwd vol %)   : train mean {d['target'].iloc[:split_idx].mean():.1f}, "
          f"test mean {d['target'].iloc[split_idx:].mean():.1f}")
    print(f"crisis-label days    : train {int(d['stress'].iloc[:split_idx].sum())}, "
          f"test {int(d['stress'].iloc[split_idx:].sum())}")
    print("\nlabeled stress windows (independent yard-stick):")
    for start, end in STRESS_WINDOWS:
        w = X.loc[(X.index >= start) & (X.index <= end)]
        if len(w) == 0:
            where = "NOT IN DATA"
        elif w.index.max() <= boundary:
            where = "train"
        elif w.index.min() >= boundary:
            where = "TEST"
        else:
            where = "straddles boundary"
        print(f"  {start} .. {end}  {len(w):4d} rows  -> {where}")
    print()
    return split_idx


# --------------------------------------------------------------------------
def single_feature_baselines(df, split_idx):
    print("=" * 74)
    print("2. SINGLE-FEATURE BASELINES  (test block = last 20%)")
    print("=" * 74)
    print("Spearman = rank corr of the raw feature with the forward-vol target.")
    print("AUC-lbl  = that raw feature as a score against the hand labels.\n")
    d = df.dropna(subset=["target"])
    tgt = d["target"].iloc[split_idx:]
    y_lbl = d["stress"].iloc[split_idx:]
    for col in FEATURE_COLUMNS:
        s = d[col].iloc[split_idx:]
        rho = spearmanr(s, tgt).statistic
        auc = roc_auc_score(y_lbl, s)
        if auc < 0.5:
            auc = roc_auc_score(y_lbl, -s)
        print(f"  {col:20s}  Spearman {rho:6.3f}   AUC-lbl {auc:5.3f}")

    # the model itself, same split
    name, est = None, None
    from model_utils import build_candidates
    meta = json.load(open("model_meta.json"))
    est = build_candidates()[meta["winner"]]
    Xtr = d[FEATURE_COLUMNS].iloc[:split_idx]
    ytr = d["target"].iloc[:split_idx]
    est.fit(Xtr, ytr)
    p = est.predict(d[FEATURE_COLUMNS].iloc[split_idx:])
    print(f"\n  {'MODEL (' + meta['winner'] + ')':20s}  Spearman {spearmanr(p, tgt).statistic:6.3f}   "
          f"AUC-lbl {roc_auc_score(y_lbl, p):5.3f}   PR-lbl {average_precision_score(y_lbl, p):5.3f}")
    print()


# --------------------------------------------------------------------------
def walk_forward_cv(df, meta, n_folds=5):
    print("=" * 74)
    print(f"3. BLOCKED WALK-FORWARD CV  ({n_folds} expanding folds)")
    print("=" * 74)
    d = df.dropna(subset=["target"])
    X = d[FEATURE_COLUMNS]
    y = d["target"]
    s = d["stress"]
    n = len(X)
    edges = np.linspace(int(n * 0.4), n, n_folds + 1, dtype=int)
    name, est = _winner_estimator(meta)
    print(f"  {'test window':<26}{'stress/tot':>12}{'Spearman':>10}{'RMSE':>8}{'AUC-lbl':>9}")
    rho_all, auc_all = [], []
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        m = clone(est)
        m.fit(X.iloc[:a], y.iloc[:a])
        p = m.predict(X.iloc[a:b])
        rho = spearmanr(p, y.iloc[a:b]).statistic
        rmse = float(np.sqrt(np.mean((p - y.iloc[a:b].to_numpy()) ** 2)))
        yy = s.iloc[a:b]
        auc = roc_auc_score(yy, p) if yy.nunique() > 1 else float("nan")
        rho_all.append(rho)
        if not np.isnan(auc):
            auc_all.append(auc)
        auc_s = f"{auc:.3f}" if not np.isnan(auc) else "   n/a"
        win = f"{X.index[a].date()} .. {X.index[b-1].date()}"
        print(f"  {win:<26}{f'{int(yy.sum())}/{len(yy)}':>12}{rho:>10.3f}{rmse:>8.2f}{auc_s:>9}")
    print(f"\n  mean Spearman {np.mean(rho_all):.3f}   mean AUC-vs-labels {np.mean(auc_all):.3f}")
    print()


# --------------------------------------------------------------------------
def leave_one_crisis_out(df, meta):
    print("=" * 74)
    print("4. LEAVE-ONE-CRISIS-OUT")
    print("=" * 74)
    print("Drop the window + 60d margin from training, fit the winner, then")
    print("score the hidden window. risk score = RiskScorer's linear-clip map")
    print("(predicted vol P25->0, P95->100), calibrated on that fold's train set.\n")
    d = df.dropna(subset=["target"])
    X = d[FEATURE_COLUMNS]
    y = d["target"]
    name, est = _winner_estimator(meta)
    print(f"  {'held-out crisis':<24}{'days':>6}{'median score':>14}{'frac>50':>9}{'AUC vs calm':>13}")
    for start, end in STRESS_WINDOWS:
        in_win = (X.index >= start) & (X.index <= end)
        if in_win.sum() == 0:
            print(f"  {start[:7]+'..'+end[:7]:<24}{'0':>6}   not in dataset")
            continue
        margin = pd.Timedelta(days=60)
        drop = (X.index >= pd.Timestamp(start) - margin) & (X.index <= pd.Timestamp(end) + margin)
        Xtr, ytr = X.loc[~drop], y.loc[~drop]
        m = clone(est)
        m.fit(Xtr, ytr)
        scorer = RiskScorer(m, FEATURE_COLUMNS, ytr)

        sc_win = scorer.score(X.loc[in_win])
        calm = X.loc[(~drop) & (d["stress"] == 0)]
        sc_calm = scorer.score(calm)
        yy = np.r_[np.ones(len(sc_win)), np.zeros(len(sc_calm))]
        ss = np.r_[sc_win, sc_calm]
        auc = roc_auc_score(yy, ss)
        print(f"  {start[:7]+'..'+end[:7]:<24}{int(in_win.sum()):>6}"
              f"{np.median(sc_win):>14.1f}{(sc_win > 50).mean():>9.2f}{auc:>13.3f}")
    print()


# --------------------------------------------------------------------------
def test_timeline(df, meta, split_idx):
    print("=" * 74)
    print("5. TEST-PERIOD RISK-SCORE TIMELINE (monthly mean)")
    print("=" * 74)
    d = df.dropna(subset=["target"])
    X = d[FEATURE_COLUMNS]
    y = d["target"]
    name, est = _winner_estimator(meta)
    m = clone(est)
    m.fit(X.iloc[:split_idx], y.iloc[:split_idx])
    scorer = RiskScorer(m, FEATURE_COLUMNS, y.iloc[:split_idx])
    p = scorer.score(X.iloc[split_idx:])
    tl = pd.DataFrame({"score": p, "label": d["stress"].iloc[split_idx:].values},
                      index=X.index[split_idx:])
    for ts, r in tl.resample("ME").mean().iterrows():
        bar = "#" * int(r["score"] / 2)
        flag = "  <-- labeled stress" if r["label"] > 0.5 else ""
        print(f"  {ts.date()}  {r['score']:5.1f} {bar}{flag}")
    print()


if __name__ == "__main__":
    df, meta = load()
    split_idx = summarise_split(df, meta)
    single_feature_baselines(df, split_idx)
    walk_forward_cv(df, meta)
    leave_one_crisis_out(df, meta)
    test_timeline(df, meta, split_idx)
