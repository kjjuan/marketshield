# MarketShield AI — Stress Model

## Setup
```
pip install -r requirements.txt
```

## Run order (needs internet access — run locally or in Colab, not in an offline sandbox)
```
python fetch_data.py     # pulls VIX, S&P 500, bonds, gold, EM from Yahoo Finance
                          # + yield curve & credit spread from FRED
                          # -> raw_market_data.csv

python train.py           # builds 9 features + the forward-vol target,
                           # runs a 3-model bake-off, saves the winner as
                           # model.pkl + model_meta.json + training_report.txt

python diagnose.py        # honest out-of-sample checks: single-feature
                           # baselines, walk-forward CV, leave-one-crisis-out

python simulate.py        # example what-if scenario (severe: VIX+25, credit+2, S&P grind)

uvicorn serve:app --reload --port 8000   # scenario API + slider UI
```

Then open **http://localhost:8000/** for the point-and-click scenario UI
(`web/index.html`), or **/docs** for the raw API. The React frontend calls
`POST /simulate` directly.

## Data notes (real-world, learned from actually running it)
- **Tickers were swapped for history depth.** The original ETFs start too
  late to cover the labeled windows: GLD 2004, AGG 2003, EEM 2003 — so the
  first pipeline run only had data from 2004-11 and the dot-com window was
  empty. Now: Bonds = `VBMFX` (1999), Gold = `GC=F` (2000-08), EM = `VEIEX`
  (1999). Dataset starts **2000-08-30**; after the 252-day trailing-high
  warm-up, features start **2001-02** — captures the 2001-2002 core of the
  dot-com bear.
- Credit-spread series changed from `BAMLH0A0HYM2` (ICE BofA HY OAS) to
  `BAA10Y` (Moody's Baa minus 10Y Treasury). As of 2024 the ICE BofA family
  on FRED is only redistributed for a rolling ~2-year window, so it can't
  supply the historical depth this model needs. `BAA10Y` is investment-grade,
  not HY, but has full history and moves in every stress episode.
- `yfinance` now returns MultiIndex columns and adjusts close by default;
  `fetch_data.py` was updated to handle that.

## Target (v3)
The v1/v2 target was the 8 hand-drawn binary stress windows. v3 trains on a
**continuous** target instead: `fwd_vol_20d` = the S&P's realized volatility
over the *next* 20 trading days (annualized %), defined in `labels.py`. This
turns ~8 crisis blocks into ~6,400 informative points, removes the arbitrary
window edges, and reframes "risk score" as *expected near-term turbulence*.
The binary windows are kept only as an **independent yard-stick** — `diagnose.py`
checks whether the regression output also separates them (AUC-vs-labels).

## Features (v2, carried into v3)
The v1 set had four 1-day % returns (S&P, bonds, gold, EM) that each scored
AUC ~0.55 in isolation — noise. v2 replaces them with slower stress signals:

| # | feature | what it is |
|---|---------|------------|
| 1 | `VIX` | level |
| 2 | `SP500_return_20d` | 20-day cumulative % return |
| 3 | `SP500_volatility` | 21-day realized vol, annualized % |
| 4 | `SP500_drawdown` | % below trailing 1-yr high (≤ 0) |
| 5 | `YieldCurveSpread` | 10Y − 3M, FRED |
| 6 | `CreditSpread` | Baa − 10Y Treasury, FRED |
| 7 | `Bonds_return_20d` | 20-day cumulative % return |
| 8 | `Gold_return_20d` | 20-day cumulative % return |
| 9 | `EM_return_20d` | 20-day cumulative % return |

## Files
- `fetch_data.py` — pulls raw data from Yahoo Finance + FRED
- `features.py` — turns raw prices into the 9 model features (see table above)
- `labels.py` — the continuous `fwd_vol_20d` target + the legacy binary stress windows (now an eval yard-stick)
- `model_utils.py` — the bake-off model field, `RiskScorer` (predicted vol → 0-100), and per-feature attribution (exact for the linear model, TreeSHAP for xgboost)
- `train.py` — builds features + target, **time-ordered** split, 3-model bake-off, picks the winner on walk-forward AUC-vs-labels, saves `model.pkl` / `model_meta.json`
- `simulate.py` — the what-if simulator: shocks to any feature (`mode="abs"` in the feature's own units — the default; `mode="pct"` multiplicative) → projected risk score + predicted fwd vol + attribution
- `serve.py` — FastAPI wrapper: `GET /baseline`, `POST /simulate`, and serves the UI at `/`. Wire the React "RUN SCENARIO" button to `POST /simulate`.
- `web/index.html` — self-contained slider UI (no build step); reference implementation for the React version
- `diagnose.py` — the reality check on model quality (see "Is the model any good?" below)
- `verify_pipeline.py` — runs the whole pipeline on synthetic random data to confirm your environment works. Ignore the metric values — they're meaningless (random data).

## What "Risk Score" means (v3)
`RiskScorer.score(row)` — the model predicts the S&P's realized vol over the
next 20 trading days, then maps that to **0–100 by a linear clip anchored on
the training-target P25 and P95**: predicted vol ≤ ~9% annualized (a calm
month) → 0; ≥ ~33% (a serious month) → 100; linear in between. It is a
turbulence estimate, not a crash forecast — keep that framing.

This calibration is deliberately *not* a percentile rank (which would force
the median day to 50). Result on real data: calm 2023–2026 months read
**5–25**, the 2022 bear reads 40–80, April 2025 (tariff selloff, unlabeled)
reads 80, the severe what-if scenario pegs at 100. Anchors are `CALIB_LO_Q` /
`CALIB_HI_Q` in `model_utils.py` if you want to retune.

A tried-and-rejected alternative: making the target a vol *ratio*
(fwd vol / trailing vol). It reads ~1.0 on calm days as hoped, but it also
reads ~1.0 *during* sustained crises (vol is already high and stays high), so
it collapsed crisis detection — walk-forward AUC-vs-labels fell from 0.88 to
0.60. Absolute forward vol + the linear-clip calibration is the better combo.

## Is the model any good? (ran `train.py` + `diagnose.py` on real data, Sep 2026)

Progression across the three iterations. "AUC-vs-labels" = does the model's
output separate the 8 independent hand-labeled crises; it's the one metric
comparable across all versions.

| metric | v1 (binary, 8f, 2004+) | v2 (binary, 9f, 2000+) | v3 (fwd-vol regression) |
|---|---|---|---|
| model | XGBoost clf | XGBoost clf | **Ridge** (won bake-off vs 2 XGB) |
| walk-forward AUC-vs-labels | 0.71 | 0.82 | **0.88** |
| worst CV fold | 0.40 *(below random)* | 0.68 | **0.81** |
| walk-forward Spearman (rank) | — | — | 0.53 |
| LOCO — 2008 / 2011 / 2020 | 0.97 / 0.75 / 0.67 | 0.97 / 0.74 / 0.73 | **1.00 / 0.97 / 0.94** *(frac of days scored >50)* |
| LOCO — **2015 / 2018 / 2022** | **0.04 / 0.23 / 0.15** | **0.03 / 0.22 / 0.30** | **0.69 / 0.88 / 0.98** |
| LOCO — dot-com | no data | 0.22 | **1.00** |

**v3 is the first version that detects the slow-grind crises** (2015-16, 2018)
when they're held out of training — the thing v1/v2 could not do. Training on
"forward volatility" instead of the hand-drawn labels generalises across
crisis *types* because it never has to memorise a specific episode's shape.

**On the score scale:** with the linear-clip calibration (see below), calm
2023–2026 months read 5–25, the 2022 bear 40–80, the severe what-if 100. The
live baseline as of Sep 2026 is ~13. (The earlier percentile-rank calibration
pinned calm days at ~50 — fixed.)

**Still true:** on the single 2022-heavy 80/20 block, raw VIX alone
(AUC-vs-labels 0.92) still edges the 9-feature model (0.90) — the
multi-feature lift only clearly shows up in the leave-one-crisis-out test,
across episodes VIX alone would miss.

`xgb_deep` (v2's tree settings, as a regressor) came dead last in the bake-off
— RMSE 10.7 vs Ridge's 5.8. Confirms the trees were over-parameterised for
this problem.

## Known limitations to disclose in your write-up
- The `fwd_vol_20d` target is realized *after* the fact — the model estimates
  it from same-day features. It is nowcasting turbulence, not forecasting a
  crash. Don't claim "predicted the crisis N days early".
- The 0–100 score is a percentile vs. the 2001–2021 training window, so it is
  only as representative as that window. A genuinely new vol regime would sit
  off the top of the calibration.
- The legacy stress windows are still hand-marked date ranges; they're now
  only used to *check* the model, but "AUC-vs-labels" inherits their fuzziness
  (e.g. is Aug 2015–Feb 2016 really one continuous stress period?).
- RMSE on the COVID fold is ~13 vol points — the model ranks the spike
  correctly but badly under-predicts its *magnitude*. Fine for a 0-100 score,
  not fine if you quote the predicted vol number directly.

## Next steps (in order)
1. ~~Fix the score floor.~~ **Done** — linear-clip calibration on P25/P95 of
   the training target (`model_utils.py`). Calm days now read 5–25.
2. **Slow-grind features** to push the 2015/2018 numbers higher still:
   % of S&P constituents above their 200-day MA (breadth), WTI crude, DXY
   (dollar), the MOVE index (bond vol).
3. Try a tiny bit of nonlinearity now that the target is continuous —
   `xgb_reg` was only 0.006 behind Ridge on walk-forward; a 2-3 feature
   interaction set might close it. Keep picking on walk-forward AUC-vs-labels.
4. Add Singapore/ASEAN layer: STI (`^STI`), USD/SGD (`SGD=X`).
5. Consider a second target head (forward max drawdown) and show both — vol
   and drawdown answer different "how bad" questions.
