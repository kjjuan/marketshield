"""
labels.py — Targets for MarketShield.

Two targets live here:

1. `add_regression_target()` — the PRIMARY target as of v3. For each day,
   the realized volatility of the S&P over the *next* `TARGET_HORIZON`
   trading days (annualized, %). This is continuous, defined on almost
   every row (~6,400 informative points instead of 8 hand-drawn blocks),
   has no arbitrary window edges, and turns "risk score" into "expected
   near-term turbulence" — an easier claim to defend than "resembles a
   period I labeled by hand".

2. `add_labels()` / `STRESS_WINDOWS` — the old binary target. Kept as an
   INDEPENDENT yard-stick: diagnose.py checks whether the regression model's
   output also separates these well-documented crises (AUC-vs-labels). It is
   no longer what the model trains on.

Windows are widely-agreed date ranges (dot-com through the 2022 selloff),
the same validate-against-known-crises approach used for published stress
indices (e.g. the St. Louis Fed Financial Stress Index).
"""

import numpy as np
import pandas as pd

TARGET_HORIZON = 20          # trading days forward
TARGET_NAME = f"fwd_vol_{TARGET_HORIZON}d"

STRESS_WINDOWS = [
    ("2000-03-01", "2002-10-15"),  # Dot-com crash
    ("2008-09-01", "2009-03-31"),  # Global Financial Crisis
    ("2010-04-15", "2010-07-01"),  # Flash crash / European debt fears begin
    ("2011-07-15", "2011-10-15"),  # US debt-ceiling / Eurozone crisis
    ("2015-08-01", "2016-02-15"),  # China slowdown / oil price crash
    ("2018-10-01", "2018-12-26"),  # Q4 2018 selloff
    ("2020-02-15", "2020-04-30"),  # COVID-19 crash
    ("2022-01-01", "2022-10-15"),  # 2022 rate-hike bear market
]


def add_regression_target(
    feats: pd.DataFrame, raw: pd.DataFrame, horizon: int = TARGET_HORIZON
) -> pd.DataFrame:
    """Add column `target` = annualized realized vol of the S&P over the next
    `horizon` trading days. Uses raw['SP500'] (feats has no price column).

    The last `horizon` rows get NaN (no full forward window) and should be
    dropped by the caller for training — but they are still scoreable at
    predict time, which is what the live "today" row needs.
    """
    out = feats.copy()
    logret = np.log(raw["SP500"]).diff()
    # rolling std at index j covers logret[j-horizon+1 .. j]; shift back by
    # `horizon` so row t carries the std of logret[t+1 .. t+horizon]
    fwd_std = logret.rolling(horizon).std().shift(-horizon)
    out["target"] = (fwd_std * np.sqrt(252) * 100).reindex(out.index)
    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a binary 'stress' column (1 = inside a labeled stress window).
    Independent evaluation yard-stick only — not a training target in v3."""
    df = df.copy()
    df["stress"] = 0
    for start, end in STRESS_WINDOWS:
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "stress"] = 1
    return df
