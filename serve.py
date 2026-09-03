"""
serve.py — Minimal FastAPI wrapper around simulate.py for the React frontend.

Run:
    uvicorn serve:app --reload --port 8000

Endpoints:
    GET  /                    -> friendly slider UI  (web/index.html)
    GET  /docs                -> raw interactive API docs (Swagger)
    GET  /health              -> liveness
    GET  /baseline            -> current feature row + current risk score
    POST /simulate            -> apply shocks, return projected score + attribution

model.pkl + model_meta.json + raw_market_data.csv must exist — run
fetch_data.py then train.py first.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from features import FEATURE_COLUMNS, build_features
from simulate import load_scorer, score_scenario

HERE = Path(__file__).parent
MODEL_PKL = HERE / "model.pkl"
META_JSON = HERE / "model_meta.json"
RAW_CSV = HERE / "raw_market_data.csv"
INDEX_HTML = HERE / "web" / "index.html"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for p in (MODEL_PKL, META_JSON, RAW_CSV):
        if not p.exists():
            raise RuntimeError(f"{p.name} not found — run fetch_data.py then train.py")
    meta = json.load(open(META_JSON))
    raw = pd.read_csv(RAW_CSV, index_col=0, parse_dates=True)
    feats = build_features(raw)
    _state["scorer"] = load_scorer(str(MODEL_PKL), str(META_JSON))
    _state["current_row"] = feats.iloc[-1]
    _state["as_of"] = str(feats.index[-1].date())
    _state["background"] = feats[FEATURE_COLUMNS].tail(250)
    _state["meta"] = meta
    yield
    _state.clear()


app = FastAPI(title="MarketShield AI — Scenario API", version="0.3.0", lifespan=lifespan)

# The bundled UI at "/" is same-origin and needs no CORS. CORS only matters for
# a SEPARATE frontend (your React app) on another domain: list its origin(s) in
# the ALLOWED_ORIGINS env var, comma-separated. Defaults cover local dev.
_default_origins = (
    "http://localhost:3000,http://localhost:5173,"
    "http://127.0.0.1:3000,http://127.0.0.1:5173"
)
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- schemas ---------------------------------------------------------------
class SimulateRequest(BaseModel):
    shocks: Dict[str, float] = Field(
        ...,
        description="feature -> shock amount. Only include features you want to move.",
        examples=[{"VIX": 25, "CreditSpread": 2.0, "SP500_return_20d": -12,
                   "SP500_drawdown": -18}],
    )
    mode: Literal["pct", "abs"] = Field(
        "abs",
        description="'abs' = added in the feature's own units (VIX points, "
        "spread points, % return) — the sane default; 'pct' = multiplicative "
        "(+0.40 = +40%), only meaningful for the strictly-positive level "
        "features (VIX, spreads, volatility).",
    )


class SimulateResponse(BaseModel):
    as_of: str
    baseline_score: float
    projected_score: float
    delta: float
    baseline_fwd_vol: float
    projected_fwd_vol: float
    shock_mode: str
    shocks_applied: Dict[str, float]
    shocked_features: Dict[str, float]
    contributions: Dict[str, float]


# --- routes --------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    """The friendly slider UI (web/index.html)."""
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": "scorer" in _state,
        "as_of": _state.get("as_of"),
        "model": _state.get("meta", {}).get("winner"),
        "target": _state.get("meta", {}).get("target"),
    }


@app.get("/baseline")
def baseline() -> dict:
    scorer = _state["scorer"]
    row = _state["current_row"][FEATURE_COLUMNS]
    return {
        "as_of": _state["as_of"],
        "model": _state["meta"]["winner"],
        "target": _state["meta"]["target"],
        "features": FEATURE_COLUMNS,
        "values": {c: float(row[c]) for c in FEATURE_COLUMNS},
        "risk_score": round(scorer.score_one(row), 1),
        "predicted_fwd_vol": round(float(scorer.predict_target(row.to_frame().T)[0]), 1),
    }


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> dict:
    unknown = [f for f in req.shocks if f not in FEATURE_COLUMNS]
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"unknown feature(s) {unknown}; valid: {FEATURE_COLUMNS}"
        )
    try:
        result = score_scenario(
            _state["scorer"], _state["current_row"], req.shocks,
            mode=req.mode, background=_state["background"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"as_of": _state["as_of"], **result}


if __name__ == "__main__":
    import uvicorn

    # host/port from env so the same file works locally and on a host (Render,
    # Railway, Fly, HF Spaces all set $PORT)
    uvicorn.run(
        "serve:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
