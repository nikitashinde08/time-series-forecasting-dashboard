"""Time-series cross-validation engine (expanding / sliding window) with per-fold forecasts."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit

from .models import evaluate, safe_forecast


@dataclass
class SplitSpec:
    fold: int
    tr_start: int
    tr_end: int   # exclusive
    te_start: int
    te_end: int   # exclusive


@dataclass
class FoldResult:
    spec: SplitSpec
    ok: bool = True
    error: str = ""
    actual: Optional[pd.Series] = None
    pred: Optional[pd.Series] = None
    lo: Optional[pd.Series] = None
    hi: Optional[pd.Series] = None
    metrics: dict = field(default_factory=dict)
    seconds: float = 0.0


def make_splits(n: int, n_splits: int, test_size: int, gap: int, scheme: str, window: int) -> list[SplitSpec]:
    """Generate rolling-origin splits. scheme: 'Expanding window' | 'Sliding window'."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap,
                           max_train_size=window if scheme == "Sliding window" else None)
    return [SplitSpec(i + 1, int(tr.min()), int(tr.max()) + 1, int(te.min()), int(te.max()) + 1)
            for i, (tr, te) in enumerate(tscv.split(np.arange(n)))]


def feasible(n: int, n_splits: int, test_size: int, gap: int, min_train: int = 40) -> tuple[bool, str]:
    need = n_splits * test_size + gap + min_train
    if n < need:
        return False, f"Need ≥ {need} observations for {n_splits} folds × {test_size} + gap {gap} + {min_train} min-train (have {n})."
    return True, ""


@st.cache_data(show_spinner=False)
def run_cv(name: str, y: pd.Series, n_splits: int, test_size: int, gap: int, scheme: str, window: int,
           params: dict) -> list[FoldResult]:
    """Fit/forecast every fold. Each fold has its own error boundary so one failure doesn't kill the run."""
    out: list[FoldResult] = []
    for sp in make_splits(len(y), n_splits, test_size, gap, scheme, window):
        fr, t0 = FoldResult(sp), time.time()
        try:
            train, test = y.iloc[sp.tr_start:sp.tr_end], y.iloc[sp.te_start:sp.te_end]
            m, lo, hi = safe_forecast(name, train, test.index, params)
            fr.actual = test
            fr.pred, fr.lo, fr.hi = (pd.Series(a, index=test.index) for a in (m, lo, hi))
            fr.metrics = evaluate(test.values, m)
        except Exception as e:  # noqa: BLE001
            fr.ok, fr.error = False, f"{type(e).__name__}: {e}"
        fr.seconds = time.time() - t0
        out.append(fr)
    return out


def summarize(folds: list[FoldResult], y: pd.Series) -> pd.DataFrame:
    rows = []
    for f in folds:
        if not f.ok:
            continue
        s = f.spec
        rows.append({"Fold": s.fold, "Train obs": s.tr_end - s.tr_start, "Test obs": s.te_end - s.te_start,
                     "Test start": y.index[s.te_start].date(), "Test end": y.index[s.te_end - 1].date(),
                     **f.metrics, "Time (s)": round(f.seconds, 1)})
    return pd.DataFrame(rows)
