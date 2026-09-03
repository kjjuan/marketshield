"""
features.py — Turns raw price series into the model features.

v2 feature set. The v1 set used four 1-day % returns (S&P, bonds, gold, EM);
diagnose.py showed each scored AUC ~0.55 in isolation — basically noise,
because a single day's move is close to random. They're replaced here with
slower, stress-bearing signals:

  1. VIX                     -> level
  2. SP500_return_20d        -> 20-day cumulative % return  (captures the grind-down)
  3. SP500_volatility        -> 21-day rolling realized vol (annualized, %)
  4. SP500_drawdown          -> % below trailing 1-yr high  (the core stress signal)
  5. YieldCurveSpread        -> level (10Y - 3M, from FRED)
  6. CreditSpread            -> level (Baa corp - 10Y Treasury, from FRED)
  7. Bonds_return_20d        -> 20-day cumulative % return of the bond fund
  8. Gold_return_20d         -> 20-day cumulative % return of gold
  9. EM_return_20d           -> 20-day cumulative % return of EM equities

All windows look backward only, so no look-ahead leakage. The 252-day
trailing high uses min_periods=120 so the series doesn't lose a full year
at the start.
"""

import numpy as np
import pandas as pd

RET_WINDOW = 20        # trading days for cumulative-return features
VOL_WINDOW = 21        # trading days for realized vol
PEAK_WINDOW = 252      # trading days (~1y) for the drawdown reference high
PEAK_MIN_PERIODS = 120

FEATURE_COLUMNS = [
    "VIX",
    "SP500_return_20d",
    "SP500_volatility",
    "SP500_drawdown",
    "YieldCurveSpread",
    "CreditSpread",
    "Bonds_return_20d",
    "Gold_return_20d",
    "EM_return_20d",
]


def _cum_return(price: pd.Series, window: int) -> pd.Series:
    """Trailing `window`-day cumulative return, in percent."""
    return (price / price.shift(window) - 1.0) * 100


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)

    df["VIX"] = raw["VIX"]

    sp = raw["SP500"]
    df["SP500_return_20d"] = _cum_return(sp, RET_WINDOW)
    df["SP500_volatility"] = (
        sp.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(252) * 100
    )
    trailing_high = sp.rolling(PEAK_WINDOW, min_periods=PEAK_MIN_PERIODS).max()
    df["SP500_drawdown"] = (sp / trailing_high - 1.0) * 100  # <= 0

    df["YieldCurveSpread"] = raw["YieldCurveSpread"]
    df["CreditSpread"] = raw["CreditSpread"]

    df["Bonds_return_20d"] = _cum_return(raw["Bonds"], RET_WINDOW)
    df["Gold_return_20d"] = _cum_return(raw["Gold"], RET_WINDOW)
    df["EM_return_20d"] = _cum_return(raw["EM"], RET_WINDOW)

    df = df.dropna()
    return df[FEATURE_COLUMNS]
