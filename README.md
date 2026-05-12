# Time Series Forecasting Studio

Streamlit app for **portfolio-grade** time series work: baselines, classical models (ARIMA / SARIMA / ETS), gradient boosting (XGBoost), **frequency-aware seasonality**, optional **walk-forward** back-testing, lightweight **diagnostics**, approximate **forecast intervals**, CSV export, and an optional **OpenAI** narrative that only receives **aggregated metrics** (no raw series rows by default).

---

## Features

- **Data**: CSV / Excel upload or built-in weekly demo data; templates for download.
- **Models**: Naive, Seasonal naive, **Drift** (linear trend from first to last point), moving average, ARIMA, SARIMA, ETS, XGBoost (lag features + recursive horizon).
- **Validation**: Single chronological hold-out, or **walk-forward** averaging over three windows when the series is long enough.
- **Metrics**: RMSE, MAE, MAPE on back-test predictions; leaderboard with automatic “best by RMSE”.
- **Charts**: Plotly history + forecast; optional illustrative interval fan (residual-based).
- **Diagnostics**: Missing counts, duplicate timestamps, simple ACF bar chart.
- **Suggestions**: Rule-based analyst bullets (always on, no API); optional **OpenAI** Markdown report (one call per click) using metrics + those bullets—`OPENAI_API_KEY` / `OPENAI_MODEL` in **Streamlit secrets**.

---

## Sample data format

| Date       | Sales |
|------------|-------|
| 2022-01-01 | 100   |
| 2022-01-08 | 115   |

One date column plus numeric columns. Sorting is applied in the app.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit secrets (OpenAI)

In Streamlit Community Cloud, add in **App secrets**:

```toml
OPENAI_API_KEY = "sk-..."
# optional — defaults to gpt-4o-mini
OPENAI_MODEL = "gpt-4o-mini"
```

Locally, use `.streamlit/secrets.toml` with the same keys (do not commit secrets).

---

## Snapshot of original code

Pre-refactor copies live in `intial_data/` under **non-conflicting names** so they are never mistaken for the live app:

- `intial_data/app_snapshot_original.py`
- `intial_data/forecasting_models_snapshot_original.py`
- plus `intial_data/requirements.txt` and `intial_data/README.md`

Always run **`streamlit run app.py`** from the **repository root** (the folder that contains the main `app.py` next to `forecasting_models.py`).

---

## Limitations (explicit)

- Intervals are **illustrative**, not calibrated prediction intervals.
- Fixed ARIMA/SARIMA orders are a pragmatic default; production systems would tune orders or use automated selection.
- Walk-forward requires sufficient history; short series fall back to a single hold-out split.
