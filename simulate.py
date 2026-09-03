"""
simulate.py — What-if stress simulator (v3).

Takes the current (most recent) feature row plus a dict of shocks, applies
them, and re-scores with the trained forward-vol model. `score` here is the
0-100 risk score = percentile of predicted next-20-day S&P vol vs. history
(see model_utils.RiskScorer). Wrapped by serve.py for the React frontend.
"""

import json

import joblib
import pandas as pd

from features import FEATURE_COLUMNS
from model_utils import RiskScorer, contributions

MODEL_PKL = "model.pkl"
META_JSON = "model_meta.json"


def load_scorer(model_pkl=MODEL_PKL, meta_json=META_JSON) -> RiskScorer:
    model = joblib.load(model_pkl)
    meta = json.load(open(meta_json))
    return RiskScorer.from_calib(model, meta["risk_calibration"])


def apply_shock(current_row: pd.Series, shocks: dict, mode: str = "abs") -> pd.Series:
    """
    shocks: dict like {"VIX": 25, "CreditSpread": 2.0, "SP500_return_20d": -12}

    mode="abs" (default): values are added in the feature's own units — VIX
        points, spread points, % return, % drawdown. This is the sane default
        for this feature set (several features pass through 0).
    mode="pct": values are multiplicative (+0.40 = +40%). Only meaningful for
        the strictly-positive level features (VIX, the two spreads, volatility).
    Only include the features you want to shock; others stay as-is.
    """
    if mode not in ("pct", "abs"):
        raise ValueError("mode must be 'pct' or 'abs'")
    shocked = current_row.copy()
    for feature, amount in shocks.items():
        if feature not in FEATURE_COLUMNS:
            raise ValueError(f"Unknown feature '{feature}'. Must be one of {FEATURE_COLUMNS}")
        if mode == "pct":
            shocked[feature] = shocked[feature] * (1 + amount)
        else:
            shocked[feature] = shocked[feature] + amount
    return shocked


def score_scenario(scorer: RiskScorer, current_row: pd.Series, shocks: dict,
                   mode: str = "abs", background: pd.DataFrame | None = None) -> dict:
    base_row = current_row[FEATURE_COLUMNS]
    shocked_row = apply_shock(base_row, shocks, mode=mode)

    baseline_score = scorer.score_one(base_row)
    shocked_score = scorer.score_one(shocked_row)
    baseline_vol = float(scorer.predict_target(base_row.to_frame().T)[0])
    shocked_vol = float(scorer.predict_target(shocked_row.to_frame().T)[0])

    if background is None:
        background = shocked_row.to_frame().T
    contrib = contributions(scorer.model, shocked_row, background)

    return {
        "baseline_score": round(baseline_score, 1),
        "projected_score": round(shocked_score, 1),
        "delta": round(shocked_score - baseline_score, 1),
        "baseline_fwd_vol": round(baseline_vol, 1),
        "projected_fwd_vol": round(shocked_vol, 1),
        "shock_mode": mode,
        "shocks_applied": shocks,
        "shocked_features": {k: float(shocked_row[k]) for k in shocks},
        "contributions": {k: round(v, 4) for k, v in contrib.items()},
    }


def get_current_row(raw_csv="raw_market_data.csv") -> pd.Series:
    """Most recent feature row, built from the fetched raw data."""
    from features import build_features

    raw = pd.read_csv(raw_csv, index_col=0, parse_dates=True)
    feats = build_features(raw)
    row = feats.iloc[-1].copy()
    row.name = str(feats.index[-1].date())
    return row


if __name__ == "__main__":
    # A "severe scenario": vol spike + credit blowout + a month-long ~12% S&P
    # slide that drags the index 18% below its 1-yr high.
    from features import build_features

    scorer = load_scorer()
    raw = pd.read_csv("raw_market_data.csv", index_col=0, parse_dates=True)
    feats = build_features(raw)
    current_row = feats.iloc[-1]

    example_shocks = {
        "VIX": 25,
        "CreditSpread": 2.0,
        "SP500_return_20d": -12,
        "SP500_drawdown": -18,
        "SP500_volatility": 25,
    }

    result = score_scenario(scorer, current_row, example_shocks, mode="abs",
                            background=feats[FEATURE_COLUMNS].tail(250))
    print(json.dumps(result, indent=2))
