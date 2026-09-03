"""
fetch_data.py — Pulls raw market data for MarketShield AI.

Requires internet access (this will NOT run inside a sandboxed/offline
environment — run it locally, in Colab, or wherever you have internet).

Sources:
  - Yahoo Finance (via yfinance): VIX, S&P 500, bonds, gold, emerging markets
  - FRED (via pandas_datareader): yield curve spread (T10Y3M),
    corporate credit spread (BAA10Y — Moody's Baa yield minus 10Y Treasury)

Ticker choices — history depth matters more than "is it the ETF everyone
quotes", because the labeled stress windows go back to 2000:
  - Bonds: VBMFX (Vanguard Total Bond, data from 1999) instead of AGG (2003)
  - Gold:  GC=F  (COMEX gold futures, data from 2000-08) instead of GLD (2004)
  - EM:    VEIEX (Vanguard Emerging Markets, data from 1999) instead of EEM (2003)
The binding constraint is now GC=F at 2000-08-30, so the dataset starts then
and captures the 2001-2002 core of the dot-com bear (not the early-2000 top).

Credit-spread series: the original design used BAMLH0A0HYM2 (ICE BofA High
Yield OAS). As of 2024 the ICE BofA family on FRED is only redistributed for
a rolling ~2-year window, so it can't supply the history this model needs.
BAA10Y is a freely-redistributable investment-grade credit-stress proxy with
full history. IG not HY, so lower absolute levels, but it moves in every
stress episode (2008, 2011, 2020, 2022).
"""

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

START_DATE = "2000-01-01"


def fetch_raw_data(start=START_DATE, end=None):
    """Fetch all raw series and align them into a single daily DataFrame."""

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    # --- Yahoo Finance tickers ---
    tickers = {
        "VIX": "^VIX",
        "SP500": "^GSPC",
        "Bonds": "VBMFX",  # Vanguard Total Bond Market (long history; AGG only from 2003)
        "Gold": "GC=F",    # COMEX gold futures (GLD only from 2004)
        "EM": "VEIEX",     # Vanguard Emerging Markets (EEM only from 2003)
    }

    yf_frames = {}
    for name, ticker in tickers.items():
        df = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True
        )
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {ticker} ({name})")
        # modern yfinance returns MultiIndex columns (Price, Ticker) even for a
        # single ticker — flatten to the price level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # auto_adjust=True means "Close" is already adjusted; older data had "Adj Close"
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        series = df[close_col].copy()
        series.name = name
        yf_frames[name] = series

    yf_df = pd.concat(yf_frames.values(), axis=1, sort=False)
    yf_df.columns = list(yf_frames.keys())

    # --- FRED series ---
    fred_series = {
        "YieldCurveSpread": "T10Y3M",   # 10Y minus 3M treasury yield
        "CreditSpread": "BAA10Y",       # Moody's Baa corp yield minus 10Y Treasury
    }

    fred_frames = {}
    for name, series_id in fred_series.items():
        fred_frames[name] = pdr.DataReader(series_id, "fred", start, end)[series_id].rename(name)

    fred_df = pd.concat(fred_frames.values(), axis=1, sort=False)
    fred_df.columns = list(fred_frames.keys())

    # --- Merge on date, forward-fill FRED series (lower frequency / holidays) ---
    raw = yf_df.join(fred_df, how="left")
    raw = raw.sort_index().ffill().dropna()

    return raw


if __name__ == "__main__":
    data = fetch_raw_data()
    data.to_csv("raw_market_data.csv")
    print(f"Fetched {len(data)} rows, {data.columns.tolist()}")
    print(f"Date range: {data.index.min().date()} -> {data.index.max().date()}")
    print(data.tail())
