from __future__ import annotations

import io
import json
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting_models import get_model_registry, infer_seasonal_period

PAGE_STYLE = """
<style>
  .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1200px; }
  h1 { font-weight: 650; letter-spacing: -0.02em; }
  div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
  .muted { color: #5f6368; font-size: 0.95rem; }
  .callout {
    border: 1px solid #e6e8eb;
    border-radius: 10px;
    padding: 14px 16px;
    background: #fafbfc;
  }
</style>
"""


def _infer_freq_token(idx: pd.DatetimeIndex) -> str | None:
    f = pd.infer_freq(idx)
    if f is not None:
        return f
    if len(idx) < 2:
        return None
    delta = pd.Series(idx).diff().median()
    if pd.isna(delta) or delta is None:
        return None
    return pd.tseries.frequencies.to_offset(delta).freqstr


def compute_future_dates(idx: pd.DatetimeIndex, horizon: int) -> tuple[pd.DatetimeIndex, str | None]:
    freq = _infer_freq_token(idx)
    last = idx[-1]
    if freq is None:
        if len(idx) < 2:
            future = pd.date_range(start=last + pd.Timedelta(days=1), periods=horizon, freq="D")
            return future, None
        step = pd.Series(idx).diff().median()
        if pd.isna(step) or step is None or step == pd.Timedelta(0):
            step = pd.Timedelta(days=1)
        future = pd.date_range(start=last + step, periods=horizon, freq=step)
        return future, None
    off = pd.tseries.frequencies.to_offset(freq)
    future = pd.date_range(start=last + off, periods=horizon, freq=freq)
    return future, freq


def simple_acf(y: pd.Series, max_lag: int = 24) -> tuple[list[int], list[float]]:
    yv = np.asarray(y.dropna().values, dtype=float)
    if len(yv) < 4:
        return [], []
    yv = yv - yv.mean()
    n = len(yv)
    var0 = float(np.dot(yv, yv) / n + 1e-12)
    acf = [1.0]
    for k in range(1, min(max_lag + 1, n - 1)):
        acf.append(float(np.dot(yv[:-k], yv[k:]) / n / var0))
    return list(range(len(acf))), acf


@st.cache_data(show_spinner=False)
def run_forecasting(
    series: pd.Series,
    horizon: int,
    compare_all: bool,
    single_model_choice: str,
    seasonal_period: int,
    walk_forward_folds: int,
):
    registry = get_model_registry()
    wf = walk_forward_folds if walk_forward_folds > 1 else 1

    def run_one(name: str):
        cls = registry[name]
        inst = cls(series, steps=horizon, seasonal_period=seasonal_period, walk_forward_folds=wf)
        return inst.fit_forecast()

    failures: list[tuple[str, str]] = []
    results = []
    forecasts: dict[str, pd.Series] = {}
    extras_map: dict[str, dict] = {}

    names = list(registry.keys()) if compare_all else [single_model_choice]

    for name in names:
        if name not in registry:
            failures.append((name, "Unknown model"))
            continue
        try:
            metrics, forecast, extras = run_one(name)
            results.append(
                {
                    "Model": name,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                    "Validation": extras.get("validation", ""),
                }
            )
            forecasts[name] = forecast.reset_index(drop=True)
            extras_map[name] = extras
        except Exception as e:
            failures.append((name, str(e)))

    if not results:
        empty = pd.DataFrame(columns=["Model", "RMSE", "MAE", "MAPE", "Validation"])
        return empty, pd.Series(dtype=float), None, {}, {}, failures

    comp_df = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_name = str(comp_df.iloc[0]["Model"])
    best_forecast = forecasts[best_name]
    best_metrics = comp_df.iloc[0]
    best_extras = extras_map[best_name]
    return comp_df, best_forecast, best_name, best_metrics, best_extras, failures


def _secrets_openai_key() -> str | None:
    try:
        v = st.secrets.get("OPENAI_API_KEY")
        return str(v).strip() if v else None
    except Exception:
        return None


def _secrets_openai_model() -> str:
    try:
        m = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
        return str(m).strip() or "gpt-4o-mini"
    except Exception:
        return "gpt-4o-mini"


def generate_ai_insight_report(
    comp_df: pd.DataFrame,
    best_name: str,
    horizon: int,
    inferred_freq: str | None,
    seasonal_period: int,
    validation_mode: str,
    n_points: int,
    target: str,
) -> str | None:
    key = _secrets_openai_key()
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        client = OpenAI(api_key=key)
        model = _secrets_openai_model()

        payload = {
            "leaderboard": comp_df[["Model", "RMSE", "MAE", "MAPE", "Validation"]].to_dict(orient="records"),
            "selected_best_by_app": best_name,
            "forecast_horizon": horizon,
            "inferred_frequency": inferred_freq,
            "seasonal_period_used": seasonal_period,
            "validation_mode": validation_mode,
            "series_length": n_points,
            "target_column": target,
        }

        system = textwrap.dedent(
            """You are a senior forecasting analyst. You only interpret the JSON metrics provided.
        Do not invent numbers. If the data or metrics are insufficient for a claim, say so briefly.
        Output Markdown with these sections exactly: ## Data quality & risks, ## Model comparison, ## Recommendation, ## Next steps.
        Keep the full response under 700 words. Be direct and professional."""
        )
        user = "Here is JSON from a forecasting app (no raw time series is included):\n" + json.dumps(
            payload, indent=2
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.25,
            max_tokens=900,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


st.set_page_config(
    page_title="Forecasting Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    forecast_horizon = st.slider("Forecast horizon (periods)", 3, 52, 12)
    compare_all = st.checkbox("Compare all models", value=True)
    registry = get_model_registry()
    model_choice = st.selectbox(
        "Single model (when comparison is off)",
        list(registry.keys()),
        disabled=compare_all,
    )
    st.markdown("---")
    use_walk_forward = st.checkbox(
        "Walk-forward validation (3 chronological windows)",
        value=False,
        help="Averages back-test metrics over multiple train/test cut points when the series is long enough.",
    )
    walk_forward_folds = 3 if use_walk_forward else 1
    show_intervals = st.checkbox(
        "Show approximate forecast intervals",
        value=True,
        help="Fan chart based on back-test residual dispersion (illustrative, not a calibrated prediction interval).",
    )
    st.markdown("---")
    st.markdown("**Data expectations**")
    st.caption("One date column, one or more numeric series, sorted chronologically.")
    st.caption("Results are cached for responsiveness.")

st.title("Time Series Forecasting Studio")
st.markdown(
    '<p class="muted">Baselines + classical models + gradient boosting, with optional walk-forward '
    "metrics and a compact AI narrative (OpenAI key via Streamlit secrets).</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="callout"><b>How to read this app:</b> start with the leaderboard, confirm baselines vs '
    "complex models, inspect diagnostics, then review the chart and export the forecast CSV.</div>",
    unsafe_allow_html=True,
)


@st.cache_data
def get_sample_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2019-01-01", periods=260, freq="W")
    trend = np.linspace(200, 500, 260)
    seasonality = 40 * np.sin(np.linspace(0, 20 * np.pi, 260))
    noise = rng.normal(0, 25, 260)
    spikes = rng.choice([0.0, 120.0], size=260, p=[0.95, 0.05])
    values = trend + seasonality + noise + spikes
    return pd.DataFrame(
        {
            "Date": dates,
            "Sales": values,
            "Revenue": values * 8 + rng.normal(0, 200, 260),
            "Orders": values / 5 + rng.normal(0, 5, 260),
        }
    )


sample_df = get_sample_data()
template_df = pd.DataFrame(
    {
        "Date": pd.date_range(start="2020-01-01", periods=52, freq="W"),
        "Sales": np.random.default_rng(1).integers(100, 200, 52),
        "Revenue": np.random.default_rng(2).integers(1000, 5000, 52),
        "Orders": np.random.default_rng(3).integers(20, 80, 52),
    }
)
csv_template = template_df.to_csv(index=False).encode("utf-8")
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    template_df.to_excel(writer, index=False)
excel_template = excel_buffer.getvalue()

st.subheader("Data")
c1, c2 = st.columns(2)
c1.download_button("Template (CSV)", csv_template, "forecast_template.csv", "text/csv")
c2.download_button("Template (Excel)", excel_template, "forecast_template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

use_sample = st.checkbox("Use built-in demo dataset", value=True)
if use_sample:
    df = sample_df.copy()
    st.info("Synthetic weekly series with trend, seasonality, and occasional spikes.")
else:
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded is None:
        st.stop()
    try:
        df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

date_col = next(
    (col for col in df.columns if pd.to_datetime(df[col], errors="coerce").notna().mean() > 0.7),
    None,
)
if date_col is None:
    st.error("No reliable date column found. Use the template format.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

numeric_cols = df.select_dtypes(include="number").columns
if len(numeric_cols) == 0:
    st.error("No numeric columns found.")
    st.stop()

target_column = st.selectbox("Target column", list(numeric_cols))
series = df[target_column].dropna().astype(float)
if len(series) < 20:
    st.warning("Short series: metrics can be noisy; interpret with caution.")

with st.expander("Preview"):
    st.dataframe(df.head(12), use_container_width=True)

future_dates, inferred_freq = compute_future_dates(series.index, forecast_horizon)
seasonal_period = infer_seasonal_period(inferred_freq, len(series))

meta_cols = st.columns(4)
meta_cols[0].metric("Observations", f"{len(series):,}")
meta_cols[1].metric("Inferred frequency", inferred_freq or "median spacing")
meta_cols[2].metric("Season length (m)", f"{seasonal_period}")
meta_cols[3].metric("Validation", "Walk-forward" if walk_forward_folds > 1 else "Hold-out")

with st.expander("Diagnostics", expanded=False):
    d1, d2 = st.columns(2)
    missing = int(series.isna().sum())
    d1.write(f"**Missing values (target):** {missing}")
    dupes = int(series.index.duplicated().sum())
    d2.write(f"**Duplicate timestamps:** {dupes}")
    if len(series) >= 8:
        lags, acf_vals = simple_acf(series, max_lag=min(36, len(series) - 2))
        if lags:
            fig_a = go.Figure(
                data=go.Bar(x=lags, y=acf_vals, marker_color="#1f77b4"),
            )
            fig_a.update_layout(
                template="plotly_white",
                height=320,
                title="Autocorrelation (ACF, simple)",
                xaxis_title="Lag",
                yaxis_title="ACF",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_a, use_container_width=True)
    st.caption(
        "ACF is a lightweight diagnostic for seasonality/trend memory. It is not a substitute for full residual analysis."
    )

st.divider()
st.subheader("Model results")

with st.spinner("Fitting models…"):
    comp_df, best_forecast, best_model_name, best_metrics, best_extras, failures = run_forecasting(
        series,
        forecast_horizon,
        compare_all,
        model_choice,
        seasonal_period,
        walk_forward_folds,
    )

if failures:
    with st.expander("Model fit notes", expanded=False):
        for name, msg in failures:
            st.caption(f"**{name}:** {msg}")

if best_model_name is None or len(comp_df) == 0:
    st.error("All models failed to fit on this series. Try a longer history or a different target.")
    st.stop()

st.success(f"Best model by hold-out / walk-forward RMSE: **{best_model_name}**")
m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{float(best_metrics['RMSE']):.3f}")
m2.metric("MAE", f"{float(best_metrics['MAE']):.3f}")
m3.metric("MAPE (%)", f"{float(best_metrics['MAPE']):.2f}")

if compare_all:
    st.markdown("**Leaderboard**")
    show_df = comp_df.copy()
    st.dataframe(
        show_df.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE": "{:.2f}"}),
        use_container_width=True,
    )

st.divider()
st.subheader("AI insight report (optional)")
if not _secrets_openai_key():
    st.caption(
        "Add `OPENAI_API_KEY` to Streamlit secrets to enable one-click narrative insights. "
        "Optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`). Only aggregated metrics are sent—no raw rows."
    )
else:
    if st.button("Generate AI insight report", type="primary"):
        with st.spinner("Calling OpenAI (single request)…"):
            report = generate_ai_insight_report(
                comp_df,
                best_model_name,
                forecast_horizon,
                inferred_freq,
                seasonal_period,
                str(best_extras.get("validation", "")),
                len(series),
                target_column,
            )
        if report:
            st.session_state["ai_report"] = report
        else:
            st.warning("Could not generate a report (missing `openai` package or API error).")

    if st.session_state.get("ai_report"):
        st.markdown(st.session_state["ai_report"])

st.divider()
st.subheader("Forecast chart")

vals = np.asarray(best_forecast, dtype=float).ravel()
if len(vals) != len(future_dates):
    future_dates = compute_future_dates(series.index, forecast_horizon)[0]
    vals = np.asarray(best_forecast, dtype=float).ravel()[: len(future_dates)]
    if len(vals) < len(future_dates):
        vals = np.pad(vals, (0, len(future_dates) - len(vals)), mode="edge")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=series.index,
        y=series.values,
        name="History",
        mode="lines",
        line=dict(color="#1f77b4", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=vals,
        name=f"Forecast ({best_model_name})",
        mode="lines",
        line=dict(color="#ff7f0e", width=3, dash="dash"),
    )
)

if show_intervals:
    rs = float(best_extras.get("residual_std", 0.0) or 0.0)
    if rs > 0:
        h = np.arange(1, len(future_dates) + 1, dtype=float)
        band = 1.96 * rs * np.sqrt(h)
        upper = vals + band
        lower = vals - band
        fig.add_trace(
            go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself",
                fillcolor="rgba(255, 127, 14, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Approx. interval",
                hoverinfo="skip",
            )
        )

fig.add_vline(x=series.index[-1], line=dict(dash="dot", color="#9aa0a6"))
fig.update_layout(
    template="plotly_white",
    height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_title="Date",
    yaxis_title=target_column,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Intervals widen with horizon using a simple sqrt(h) scaling of residual dispersion; use for communication, not pricing risk."
)

st.divider()
st.subheader("Export")
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": vals})
st.dataframe(forecast_df.head(10), use_container_width=True)
csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download forecast ({best_model_name})",
    data=csv_bytes,
    file_name=f"{best_model_name.replace(' ', '_')}_forecast.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Portfolio build: baselines + classical + ML, frequency-aware seasonality, optional walk-forward metrics, "
    "and optional OpenAI narrative on aggregated results only."
)
