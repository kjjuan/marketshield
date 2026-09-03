# Deploying MarketShield

The app is a single FastAPI service (`serve.py`) that serves both the API and
the slider UI at `/`. It loads `model.pkl`, `model_meta.json` and
`raw_market_data.csv` at startup — those are committed to the repo, so the host
needs no internet access and no build step beyond `pip install`.

`serve.py` reads `$PORT` / `$HOST` from the environment, so it runs unchanged on
any Python host. `Procfile` and `runtime.txt` are included.

---

## Option A — Render (free, simplest)

1. Push this folder to a GitHub repo (see "Git setup" below).
2. Go to <https://render.com> → **New +** → **Web Service** → connect the repo.
3. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn serve:app --host 0.0.0.0 --port $PORT`
   - **Instance type:** Free
4. Create. First build takes ~3–5 min (xgboost/scipy wheels are large).
5. You get `https://<name>.onrender.com` — the UI is at `/`, API at `/docs`.

Free instances sleep after ~15 min idle; the first hit after that takes ~30 s
to wake. Fine for a demo, not for a live embed.

## Option B — Hugging Face Spaces (free, no card)

1. Create a Space → SDK: **Docker** → blank, or use the FastAPI template.
2. Push these files. Add a `Dockerfile`:
   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   ENV PORT=7860
   CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
3. Spaces exposes port 7860 automatically; the UI is at the Space URL.

## Option C — Railway / Fly.io

Both auto-detect the `Procfile`. Railway: "Deploy from GitHub repo", done.
Fly: `fly launch` (accept the Procfile), `fly deploy`.

---

## Git setup (one time)

```bash
cd marketshield
git init
git add .
git commit -m "MarketShield: deployable FastAPI app + slider UI"
git branch -M main
git remote add origin https://github.com/<you>/marketshield.git
git push -u origin main
```

`.gitignore` excludes `.venv/` and caches but **keeps** the model + data files
on purpose (the app needs them and nothing regenerates them on the host).

---

## Connecting your React frontend

If the React app is served from a **different domain** than this API, set the
`ALLOWED_ORIGINS` env var on the host to that domain, comma-separated, e.g.:

```
ALLOWED_ORIGINS = https://marketshield-ui.vercel.app,https://www.yoursite.com
```

Then in React, point fetches at the deployed API base instead of
`http://localhost:8000`. If you serve the React build from FastAPI itself
(same origin), no CORS config is needed.

---

## Refreshing the data / model

The committed `raw_market_data.csv` is a snapshot — `/baseline` will keep
showing its last date until you update it:

```bash
python fetch_data.py      # pull fresh data
python train.py           # retrain, rewrite model.pkl + model_meta.json
git add raw_market_data.csv model.pkl model_meta.json feature_stats.json training_report.txt
git commit -m "refresh data + model"
git push                  # host redeploys automatically
```

To automate, add a scheduled GitHub Action that runs those two scripts and
commits the artifacts on a cron.

---

## Notes / limitations for a public demo

- The served model is fit on the **training split** (through ~mid-2021), which
  is correct for the honest evaluation in `diagnose.py` but not ideal for a
  live "today's number". For a public demo, refit on all rows before deploying:
  in `train.py`, after picking the winner, `best.fit(X, y)` on the full set and
  re-save. (Left as-is so the eval stays honest by default.)
- No auth / rate limiting. It's a read-only inference endpoint, low risk, but
  don't put it behind a paid autoscaler without a limiter.
- `shap` is only imported if the winning model is xgboost; the current Ridge
  winner uses an exact closed-form attribution and never imports it. You can
  drop `shap` from `requirements.txt` to cut ~1 min off the build while Ridge
  is the winner.
