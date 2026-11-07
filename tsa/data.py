"""Data ingestion + preprocessing (cached)."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


def make_sample_data(n_days: int = 900, seed: int = 7) -> pd.Series:
    """Synthetic daily series: trend + weekly + yearly seasonality + noise + outliers + gaps."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    t = np.arange(n_days)
    y = 100 + 0.08 * t + 6 * np.sin(2 * np.pi * t / 7) + 12 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 2.5, n_days)
    for i in rng.choice(n_days, 8, replace=False):
        y[i] += rng.choice([-1, 1]) * rng.uniform(18, 30)
    s = pd.Series(y, index=idx, name="Sample")
    s.iloc[rng.choice(n_days, 12, replace=False)] = np.nan
    return s


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yahoo(ticker: str, period: str, field: str) -> pd.Series:
    """Download daily prices from Yahoo Finance as a single series."""
    import yfinance as yf

    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol or try a longer period.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if field not in raw.columns:
        raise ValueError(f"Field '{field}' not found. Available: {list(raw.columns)}")
    s = raw[field].astype(float).dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = f"{ticker} {field}"
    return s


def parse_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@dataclass
class PrepInfo:
    freq: str
    n_raw: int
    n_final: int
    n_missing: int
    n_duplicates: int
    method: str


@st.cache_data(show_spinner=False)
def preprocess(y: pd.Series, freq: str, method: str) -> tuple[pd.Series, PrepInfo]:
    """Sort, de-duplicate, regularise to a fixed frequency and impute gaps."""
    y = pd.to_numeric(y, errors="coerce")
    y.index = pd.to_datetime(y.index)
    y = y[~y.index.isna()].sort_index()
    n_raw = len(y)
    n_dup = int(y.index.duplicated().sum())
    y = y[~y.index.duplicated(keep="last")].asfreq(freq)
    n_missing = int(y.isna().sum())
    if method == "Linear interpolation":
        y = y.interpolate("linear")
    elif method == "Forward fill":
        y = y.ffill()
    elif method == "Rolling mean (7)":
        y = y.fillna(y.rolling(7, min_periods=1, center=True).mean())
    y = y.ffill().bfill()
    return y, PrepInfo(freq, n_raw, len(y), n_missing, n_dup, method)


def fingerprint(y: pd.Series) -> str:
    """Cheap identity of a series, used to invalidate stored tuning / CV results when data changes."""
    return f"{len(y)}|{y.index[0]:%Y%m%d}|{y.index[-1]:%Y%m%d}|{float(y.iloc[-1]):.6f}|{float(y.sum()):.4f}"
