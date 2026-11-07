"""Forecasting engines (ARIMA, SARIMA, Prophet, MLP) with a uniform interface and error boundaries."""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class ModelResult:
    name: str
    ok: bool = True
    error: str = ""
    test_pred: Optional[pd.Series] = None
    test_lo: Optional[pd.Series] = None
    test_hi: Optional[pd.Series] = None
    fut_pred: Optional[pd.Series] = None
    fut_lo: Optional[pd.Series] = None
    fut_hi: Optional[pd.Series] = None
    metrics: dict = field(default_factory=dict)
    seconds: float = 0.0


def future_index(last: pd.Timestamp, steps: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=last, periods=steps + 1, freq=freq)[1:]


# ---- individual forecasters: (train Series, target DatetimeIndex, params) -> (mean, lo, hi) ------
def _arima(train, idx, p):
    fit = ARIMA(train.values, order=(p["p"], p["d"], p["q"])).fit()
    f = fit.get_forecast(len(idx)).summary_frame(alpha=0.05)
    return f["mean"].values, f["mean_ci_lower"].values, f["mean_ci_upper"].values


def _sarima(train, idx, p):
    mod = SARIMAX(train.values, order=(p["p"], p["d"], p["q"]),
                  seasonal_order=(p["P"], p["D"], p["Q"], p["m"]),
                  enforce_stationarity=False, enforce_invertibility=False)
    fit = mod.fit(disp=False, maxiter=60)
    f = fit.get_forecast(len(idx)).summary_frame(alpha=0.05)
    return f["mean"].values, f["mean_ci_lower"].values, f["mean_ci_upper"].values


def _prophet(train, idx, p):
    try:
        from prophet import Prophet
    except Exception as e:  # missing / broken install
        raise RuntimeError(f"Prophet is not available in this environment ({e}).") from e
    for n in ("cmdstanpy", "prophet"):
        logging.getLogger(n).setLevel(logging.ERROR)
    m = Prophet(interval_width=0.95, seasonality_mode=p["mode"], changepoint_prior_scale=p["cps"],
                yearly_seasonality=p["yearly"], weekly_seasonality=p["weekly"], daily_seasonality=False)
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    fc = m.predict(pd.DataFrame({"ds": idx}))
    return fc["yhat"].values, fc["yhat_lower"].values, fc["yhat_upper"].values


def _mlp(train, idx, p):
    """Sliding-window MLP (LSTM-style) with recursive forecasting; CI = 1.96 * resid_sd * sqrt(step)."""
    lb, steps = int(p["lookback"]), len(idx)
    if len(train) < lb + 20:
        raise ValueError(f"Need at least {lb + 20} observations for lookback={lb}.")
    mu, sd = float(train.mean()), float(train.std() or 1.0)
    z = (train.values - mu) / sd
    X = np.lib.stride_tricks.sliding_window_view(z, lb)[:-1]
    yv = z[lb:]
    net = MLPRegressor(hidden_layer_sizes=(int(p["units"]), max(8, int(p["units"]) // 2)), activation="relu",
                       solver="adam", learning_rate_init=1e-3, max_iter=int(p["epochs"]), random_state=42, alpha=1e-3)
    net.fit(X, yv)
    resid_sd = float(np.std(yv - net.predict(X))) * sd
    win, preds = list(z[-lb:]), []
    for _ in range(steps):
        nxt = float(np.clip(net.predict(np.array(win[-lb:])[None, :])[0], -6, 6))
        preds.append(nxt)
        win.append(nxt)
    mean = np.array(preds) * sd + mu
    se = resid_sd * np.sqrt(np.arange(1, steps + 1))
    return mean, mean - 1.96 * se, mean + 1.96 * se


FORECASTERS = {"ARIMA": _arima, "SARIMA": _sarima, "Prophet": _prophet, "LSTM (MLP)": _mlp}


def safe_forecast(name: str, train: pd.Series, idx: pd.DatetimeIndex, params: dict):
    """Run a forecaster with warnings silenced. Raises on failure (callers decide how to handle it)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("ignore", ConvergenceWarning)
        return FORECASTERS[name](train, idx, params)


def evaluate(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    nz = actual != 0
    mape = float(np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])) * 100) if nz.any() else np.nan
    return {"RMSE": float(np.sqrt(mean_squared_error(actual, pred))), "MAE": float(mean_absolute_error(actual, pred)),
            "MAPE (%)": mape, "R²": float(r2_score(actual, pred)) if len(actual) > 1 else np.nan}


@st.cache_data(show_spinner=False)
def run_model(name: str, y: pd.Series, horizon: int, test_n: int, freq: str, params: dict) -> ModelResult:
    """Back-test on the last `test_n` points, then refit on everything and forecast `horizon` ahead. Never raises."""
    res, t0 = ModelResult(name), time.time()
    try:
        train, test = y.iloc[:-test_n], y.iloc[-test_n:]
        m, lo, hi = safe_forecast(name, train, test.index, params)
        res.test_pred, res.test_lo, res.test_hi = (pd.Series(a, index=test.index) for a in (m, lo, hi))
        res.metrics = evaluate(test.values, m)
        fidx = future_index(y.index[-1], horizon, freq)
        m2, lo2, hi2 = safe_forecast(name, y, fidx, params)
        res.fut_pred, res.fut_lo, res.fut_hi = (pd.Series(a, index=fidx) for a in (m2, lo2, hi2))
    except Exception as e:  # noqa: BLE001 - deliberate error boundary
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
    res.seconds = time.time() - t0
    return res
