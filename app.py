"""
Advanced Daily Time Series Analytics & Forecasting Explorer
Run:  streamlit run app.py        (from this folder, so .streamlit/config.toml light theme is picked up)
"""
from __future__ import annotations

import traceback

import streamlit as st

st.set_page_config(page_title="Time Series Analytics & Forecasting Explorer", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

from tsa import views  # noqa: E402
from tsa.data import preprocess  # noqa: E402
from tsa.sidebar import render_sidebar  # noqa: E402
from tsa.styles import inject_styles  # noqa: E402
from tsa.ui import callout, hero, reset_registry  # noqa: E402

inject_styles()
reset_registry()  # figures are re-registered by each tab on every run (feeds the Export tab)
hero("📈 Time Series Analytics &amp; Forecasting Explorer",
     "Ingest, diagnose, tune, cross-validate and benchmark ARIMA · SARIMA · Prophet · Neural networks — then export an executive report.",
     ["Optuna tuning", "Time-series CV", "PDF export", "Light executive UI"])

S, raw, load_error = render_sidebar()
if load_error:
    callout("error", load_error + " Switch to <b>Sample data</b> or <b>Upload CSV</b> in the sidebar.")
    st.stop()
if raw is None:
    callout("info", "Upload a CSV (date column + numeric value column) or choose another source in the sidebar.")
    st.stop()

try:
    with st.spinner("Preprocessing…"):
        y_full, prep = preprocess(raw, S.freq, S.impute)
        y = y_full.iloc[-S.max_hist:]
except Exception as e:  # noqa: BLE001
    callout("error", f"Preprocessing failed: {e}")
    st.stop()
if len(y) < 60:
    callout("error", f"Only {len(y)} observations after preprocessing; at least 60 are required.")
    st.stop()

ctx = views.AppCtx(y=y, prep=prep, S=S, test_n=int(min(S.horizon, max(7, len(y) // 4))),
                   label=str(y.name or "Value"), source_label=f"{S.source} - {raw.name}")

TABS = [("📊 Overview", views.render_overview), ("🔬 Diagnostics", views.render_diagnostics),
        ("🔮 Forecasting", views.render_forecast), ("🏆 Benchmark", views.render_benchmark),
        ("🎯 Tuning (Optuna)", views.render_tuning), ("🧪 Cross-Validation", views.render_cv), ("📦 Export", views.render_export)]
for tab, (name, fn) in zip(st.tabs([t[0] for t in TABS]), TABS):
    with tab:
        try:  # error boundary: a failure in one tab never takes down the others
            fn(ctx)
        except Exception as e:  # noqa: BLE001
            callout("error", f"<b>{name}</b> hit an unexpected error: <code>{type(e).__name__}: {e}</code>")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

st.markdown("---")
st.caption("Streamlit · Plotly · statsmodels · Prophet · scikit-learn · Optuna · ReportLab. Not financial advice.")
