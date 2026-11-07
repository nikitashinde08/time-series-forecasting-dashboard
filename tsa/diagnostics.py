"""Statistical diagnostics: ADF, descriptive stats, anomaly detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def run_adf(y: pd.Series) -> dict:
    """Augmented Dickey-Fuller. H0: unit root (non-stationary)."""
    out = adfuller(y.dropna().values, autolag="AIC")
    stat, p, lags, nobs, crit = out[0], out[1], out[2], out[3], out[4]
    return {"stat": float(stat), "p": float(p), "lags": int(lags), "nobs": int(nobs),
            "crit": {k: float(v) for k, v in crit.items()}, "stationary": bool(p < 0.05)}


def summary_table(y: pd.Series) -> pd.DataFrame:
    d = y.describe()
    out = {"Count": d["count"], "Mean": d["mean"], "Std Dev": d["std"], "Min": d["min"], "25%": d["25%"],
           "Median": d["50%"], "75%": d["75%"], "Max": d["max"], "Skewness": y.skew(), "Kurtosis": y.kurt(),
           "Coeff. of Variation": d["std"] / d["mean"] if d["mean"] else np.nan}
    return pd.DataFrame({"Value": out}).round(4)


def detect_anomalies(y: pd.Series, method: str, iqr_k: float, iqr_basis: str, window: int, z_thr: float):
    """Return (boolean mask, bands dict) for IQR or rolling Z-score detection."""
    if method == "IQR":
        med = y.rolling(window, min_periods=3, center=True).median()
        base = y if iqr_basis == "Raw values" else y - med
        q1, q3 = base.quantile(0.25), base.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        mask = (base < lo) | (base > hi)
        bands = ({"lower": pd.Series(lo, index=y.index), "upper": pd.Series(hi, index=y.index)}
                 if iqr_basis == "Raw values" else {"lower": med + lo, "upper": med + hi})
    else:
        mp = max(3, window // 2)
        mu, sd = y.rolling(window, min_periods=mp).mean(), y.rolling(window, min_periods=mp).std()
        mask = ((y - mu) / sd.replace(0, np.nan)).abs() > z_thr
        bands = {"lower": mu - z_thr * sd, "upper": mu + z_thr * sd}
    return mask.fillna(False), bands
