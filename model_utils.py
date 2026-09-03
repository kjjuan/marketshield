"""
model_utils.py — shared bits for the v3 regression pipeline.

  * build_candidates()  — the model bake-off field
  * RiskScorer          — wraps a fitted regressor + a calibration so its
                          output becomes a 0-100 "risk score" (percentile of
                          predicted forward vol vs. the training distribution)
  * contributions()     — per-feature attribution for one row, dispatching
                          on model type (exact for linear, TreeSHAP for xgboost)

Everything downstream (train / diagnose / simulate / serve) goes through
RiskScorer so the 0-100 number means the same thing everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


# --------------------------------------------------------------------------
def build_candidates() -> dict:
    """Name -> unfitted estimator. All take a raw feature DataFrame in .fit /
    .predict (the Ridge pipeline self-scales)."""
    return {
        "ridge": Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))]
        ),
        # heavily regularised trees — the honest nonlinear option
        "xgb_reg": xgb.XGBRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=20,
            reg_lambda=5.0,
            random_state=42,
        ),
        # the v2-classifier tree settings, as a regressor — kept only to show
        # in the report that it over-fits this few events
        "xgb_deep": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }


def is_linear_pipeline(model) -> bool:
    return isinstance(model, Pipeline) and isinstance(model.steps[-1][1], Ridge)


def is_xgb(model) -> bool:
    return isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier))


# --------------------------------------------------------------------------
CALIB_LO_Q = 0.25   # training-target quantile that maps to risk score 0
CALIB_HI_Q = 0.95   # training-target quantile that maps to risk score 100


class RiskScorer:
    """Fitted regressor + a monotone map from predicted forward-vol to 0-100.

    The map is LINEAR-in-value with clipping, anchored on two training-target
    quantiles: predicted vol at/below the training P25 (~10% annualized, a
    calm month) -> score 0; at/above the training P95 (~33%, a serious month)
    -> score 100. This deliberately keeps the score low most of the time
    (a percentile-rank map would force the median day to 50).
    """

    def __init__(self, model, feature_columns, y_train_target,
                 lo_q=CALIB_LO_Q, hi_q=CALIB_HI_Q):
        self.model = model
        self.feature_columns = list(feature_columns)
        y = np.asarray(y_train_target, dtype=float)
        self.lo, self.hi = (float(v) for v in np.quantile(y, [lo_q, hi_q]))

    # --- raw regression output (annualized vol %) ---
    def predict_target(self, X) -> np.ndarray:
        X = self._frame(X)
        return np.asarray(self.model.predict(X), dtype=float)

    # --- calibrated 0-100 risk score ---
    def score(self, X) -> np.ndarray:
        pred = self.predict_target(X)
        return np.clip((pred - self.lo) / (self.hi - self.lo) * 100.0, 0.0, 100.0)

    def score_one(self, row: pd.Series) -> float:
        return float(self.score(row.to_frame().T)[0])

    def _frame(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        return X[self.feature_columns]

    # --- persistence (plain dict; model pickled separately) ---
    def calib(self) -> dict:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "feature_columns": self.feature_columns,
        }

    @classmethod
    def from_calib(cls, model, calib: dict) -> "RiskScorer":
        obj = cls.__new__(cls)
        obj.model = model
        obj.feature_columns = list(calib["feature_columns"])
        obj.lo = float(calib["lo"])
        obj.hi = float(calib["hi"])
        return obj


# --------------------------------------------------------------------------
def contributions(model, row: pd.Series, background: pd.DataFrame) -> dict:
    """Per-feature attribution to the predicted forward vol for a single row,
    sorted by |impact|. Exact for the linear pipeline; TreeSHAP for xgboost;
    permutation SHAP as a fallback.
    """
    cols = list(row.index)
    x = row[cols].to_numpy(dtype=float)

    if is_linear_pipeline(model):
        scaler: StandardScaler = model.named_steps["scale"]
        ridge: Ridge = model.named_steps["ridge"]
        z = (x - scaler.mean_) / scaler.scale_
        contrib = ridge.coef_ * z  # exact additive decomposition
        vals = dict(zip(cols, (float(v) for v in contrib)))
    elif is_xgb(model):
        import shap

        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(row[cols].to_frame().T)
        vals = dict(zip(cols, (float(v) for v in np.ravel(sv))))
    else:
        import shap

        expl = shap.Explainer(model.predict, background[cols])
        sv = expl(row[cols].to_frame().T)
        vals = dict(zip(cols, (float(v) for v in np.ravel(sv.values))))

    return dict(sorted(vals.items(), key=lambda kv: abs(kv[1]), reverse=True))
