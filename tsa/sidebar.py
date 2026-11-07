"""Sidebar: data source, preprocessing, model selection + hyper-parameters, diagnostics options."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

from .config import CPS_OPTIONS, MODEL_NAMES, UNITS_OPTIONS
from .data import fetch_yahoo, make_sample_data, parse_uploaded_csv


def _sl(label: str, lo: int, hi: int, default: int, key: str, step: int = 1) -> int:
    """Slider whose default lives in session_state, so tuned values can be pushed in without Streamlit warnings."""
    st.session_state.setdefault(key, default)
    return st.slider(label, lo, hi, step=step, key=key)


@dataclass
class Settings:
    source: str
    freq: str
    impute: str
    max_hist: int
    model_choice: str
    horizon: int
    params: dict
    decomp_model: str
    decomp_period: int
    anom_method: str
    iqr_k: float
    iqr_basis: str
    z_thr: float
    roll_win: int
    lag_max: int


def render_sidebar() -> tuple[Settings, Optional[pd.Series], str]:
    """Draw the sidebar and return (settings, raw_series, load_error)."""
    raw: Optional[pd.Series] = None
    load_error, default_freq_idx = "", 0
    with st.sidebar:
        st.markdown("## ⚙️ Data source")
        source = st.radio("Source", ["Yahoo Finance (live)", "Upload CSV", "Sample data"], index=2, label_visibility="collapsed")

        if source == "Yahoo Finance (live)":
            ticker = st.text_input("Ticker", "AAPL", help="AAPL, MSFT, BTC-USD, ETH-USD, ^GSPC …").strip().upper()
            period = st.selectbox("History", ["1y", "2y", "5y", "10y", "max"], index=1)
            field_ = st.selectbox("Price field", ["Close", "Open", "High", "Low", "Volume"])
            default_freq_idx = 0 if ticker.endswith("-USD") else 1
            try:
                with st.spinner(f"Fetching {ticker}…"):
                    raw = fetch_yahoo(ticker, period, field_)
            except Exception as e:  # noqa: BLE001
                load_error = f"Could not fetch live data: {e}"
        elif source == "Upload CSV":
            up = st.file_uploader("CSV file", type=["csv"])
            if up is not None:
                try:
                    df = parse_uploaded_csv(up.getvalue())
                    dc = st.selectbox("Date column", df.columns, index=0)
                    nums = [c for c in df.columns if c != dc and pd.api.types.is_numeric_dtype(df[c])]
                    if not nums:
                        raise ValueError("No numeric value column found.")
                    vc = st.selectbox("Value column", nums)
                    s = pd.Series(df[vc].values, index=pd.to_datetime(df[dc], errors="coerce"), name=vc)
                    raw = s[s.index.notna()]
                    if raw.empty:
                        raise ValueError("Date column could not be parsed.")
                except Exception as e:  # noqa: BLE001
                    load_error = f"CSV problem: {e}"
            ex = make_sample_data().rename_axis("date").reset_index().rename(columns={"Sample": "value"})
            st.download_button("⬇️ Example CSV", ex.to_csv(index=False), "example_series.csv", "text/csv")
        else:
            raw = make_sample_data()

        st.markdown("## 🧹 Preprocessing")
        freq = "D" if st.selectbox("Frequency", ["Calendar daily (D)", "Business days (B)"], index=default_freq_idx).startswith("Calendar") else "B"
        impute = st.selectbox("Missing-value imputation", ["Linear interpolation", "Forward fill", "Rolling mean (7)"])
        max_hist = st.slider("Max history used for modelling", 200, 3000, 1000, 100, help="Most recent N observations. Smaller = faster.")

        st.markdown("## 🔮 Forecasting")
        model_choice = st.selectbox("Model", MODEL_NAMES, index=0, key="model_choice")
        horizon = st.slider("Forecast horizon (days)", 7, 120, 30)
        sd = 7 if freq == "D" else 5

        with st.expander("ARIMA parameters", expanded=model_choice == "ARIMA"):
            a_p = _sl("p (AR lags)", 0, 10, 2, "a_p")
            a_d = _sl("d (differencing)", 0, 2, 1, "a_d")
            a_q = _sl("q (MA lags)", 0, 10, 2, "a_q")
        with st.expander("SARIMA parameters", expanded=model_choice == "SARIMA"):
            s_p = _sl("p", 0, 5, 1, "s_p"); s_d = _sl("d", 0, 2, 1, "s_d"); s_q = _sl("q", 0, 5, 1, "s_q")
            s_P = _sl("P (seasonal AR)", 0, 3, 1, "s_P"); s_D = _sl("D (seasonal diff)", 0, 1, 0, "s_D")
            s_Q = _sl("Q (seasonal MA)", 0, 3, 1, "s_Q")
            s_m = (st.session_state.setdefault("s_m", sd), st.number_input("m (season length)", 2, 400, key="s_m"))[1]
        with st.expander("Prophet parameters", expanded=model_choice == "Prophet"):
            pr_mode = st.selectbox("Seasonality mode", ["additive", "multiplicative"], key="pr_mode")
            pr_cps = (st.session_state.setdefault("pr_cps", 0.05), st.select_slider("Changepoint prior scale", CPS_OPTIONS, key="pr_cps"))[1]
            pr_year = (st.session_state.setdefault("pr_year", True), st.checkbox("Yearly seasonality", key="pr_year"))[1]
            pr_week = (st.session_state.setdefault("pr_week", True), st.checkbox("Weekly seasonality", key="pr_week"))[1]
        with st.expander("LSTM-style network (MLP)", expanded=model_choice == "LSTM (MLP)"):
            nn_lb = _sl("Look-back window", 3, 90, 21, "nn_lb")
            nn_units = (st.session_state.setdefault("nn_units", 64), st.select_slider("Hidden units", UNITS_OPTIONS, key="nn_units"))[1]
            nn_ep = _sl("Epochs (max iterations)", 50, 800, 300, "nn_ep", 50)
        st.caption("💡 Use the **Tuning** tab to search these automatically with Optuna.")

        st.markdown("## 🔍 Diagnostics")
        decomp_model = st.radio("Decomposition", ["Additive", "Multiplicative"], horizontal=True)
        decomp_period = st.number_input("Seasonal period", 2, 400, sd)
        anom_method = st.radio("Anomaly method", ["IQR", "Rolling Z-Score"], horizontal=True)
        if anom_method == "IQR":
            iqr_k = st.slider("IQR multiplier", 1.0, 4.0, 1.5, 0.1)
            iqr_basis, z_thr = st.radio("Apply IQR to", ["Rolling-median residuals", "Raw values"], horizontal=True), 3.0
        else:
            z_thr = st.slider("|Z| threshold", 1.5, 5.0, 3.0, 0.1)
            iqr_k, iqr_basis = 1.5, "Raw values"
        roll_win = st.slider("Rolling window", 5, 120, 30)
        lag_max = st.slider("Max lags (ACF/PACF)", 10, 100, 40)

    params = {
        "ARIMA": {"p": a_p, "d": a_d, "q": a_q},
        "SARIMA": {"p": s_p, "d": s_d, "q": s_q, "P": s_P, "D": s_D, "Q": s_Q, "m": int(s_m)},
        "Prophet": {"mode": pr_mode, "cps": pr_cps, "yearly": pr_year, "weekly": pr_week},
        "LSTM (MLP)": {"lookback": nn_lb, "units": nn_units, "epochs": nn_ep},
    }
    S = Settings(source, freq, impute, max_hist, model_choice, horizon, params, decomp_model, int(decomp_period),
                 anom_method, iqr_k, iqr_basis, z_thr, roll_win, lag_max)
    return S, raw, load_error
