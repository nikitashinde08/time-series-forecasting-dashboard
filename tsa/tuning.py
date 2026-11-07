"""
Multi-step hyper-parameter tuning with Optuna.

Pipeline
  0. Baseline  - score the current sidebar parameters with the same CV objective.
  1. Stage 1   - broad TPE exploration over the full search space (with median pruning across CV folds).
  2. Stage 2   - the space is narrowed around the best ~30% of Stage-1 trials, the best trial is
                 re-enqueued, and a second TPE study refines it.
Objective = mean RMSE over rolling-origin (TimeSeriesSplit) folds, computed on the TRAINING portion only,
so the final hold-out used for benchmarking is never touched (no leakage).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import SEARCH_SPACES
from .models import safe_forecast


@dataclass
class TuneConfig:
    stage1_trials: int = 20
    stage2_trials: int = 12
    folds: int = 3
    cv_test_size: int = 30
    timeout_s: int = 90
    history: int = 400
    seed: int = 42
    prune: bool = True


@dataclass
class TuneResult:
    model: str
    best_params: dict
    best_value: float
    baseline_params: dict
    baseline_value: float
    trials: pd.DataFrame
    importances: dict
    space1: dict
    space2: dict
    n_complete: int
    n_pruned: int
    seconds: float
    cfg: dict = field(default_factory=dict)

    @property
    def improvement_pct(self) -> float:
        if not np.isfinite(self.baseline_value) or self.baseline_value == 0:
            return float("nan")
        return (self.baseline_value - self.best_value) / self.baseline_value * 100


def cv_rmse(name: str, y: pd.Series, params: dict, folds: int, test_size: int, trial=None) -> float:
    """Mean RMSE across rolling-origin folds. Reports per-fold to Optuna so bad trials can be pruned early."""
    import optuna

    scores = []
    for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=folds, test_size=test_size).split(y)):
        train, test = y.iloc[tr], y.iloc[te]
        pred, _, _ = safe_forecast(name, train, test.index, params)
        scores.append(float(np.sqrt(np.mean((test.values - pred) ** 2))))
        if trial is not None:
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores))


def _suggest(trial, space: dict) -> dict:
    out = {}
    for k, spec in space.items():
        out[k] = trial.suggest_int(k, spec[1], spec[2]) if spec[0] == "int" else trial.suggest_categorical(k, spec[1])
    return out


def _refine(space: dict, df: pd.DataFrame, frac: float = 0.3) -> dict:
    """Shrink the search space to the range spanned by the best `frac` of completed trials."""
    top = df.nsmallest(max(3, math.ceil(len(df) * frac)), "value")
    new = {}
    for k, spec in space.items():
        vals = top[k]
        if spec[0] == "int":
            new[k] = ("int", int(vals.min()), int(vals.max()))
        else:
            keep = [c for c in spec[1] if c in set(vals.tolist())]
            new[k] = ("cat", keep or spec[1])
    return new


def _records(study, stage: int, offset: int) -> list[dict]:
    rows = []
    for t in study.trials:
        rows.append({"trial": offset + t.number, "stage": stage, "state": t.state.name,
                     "value": t.value, "seconds": (t.duration.total_seconds() if t.duration else np.nan), **t.params})
    return rows


def run_tuning(name: str, y: pd.Series, base_params: dict, cfg: TuneConfig,
               progress: Optional[Callable[[float, str], None]] = None) -> TuneResult:
    """Execute the full multi-stage study. Raises on unrecoverable errors (caller shows a banner)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    y = y.iloc[-cfg.history:]
    if len(y) < cfg.cv_test_size * (cfg.folds + 1):
        raise ValueError(f"Series too short for {cfg.folds} folds of {cfg.cv_test_size} points "
                         f"(have {len(y)}, need ≥ {cfg.cv_test_size * (cfg.folds + 1)}). Reduce folds or the test window.")
    space1 = SEARCH_SPACES[name]
    tunable = {k: base_params[k] for k in space1 if k in base_params}
    total = cfg.stage1_trials + cfg.stage2_trials
    done = {"n": 0}

    def tick(label: str):
        def cb(study, trial):
            done["n"] += 1
            best = study.best_value if any(t.value is not None for t in study.trials) else float("nan")
            if progress:
                progress(min(done["n"] / max(total, 1), 1.0), f"{label}: trial {done['n']}/{total} · best CV-RMSE {best:.3f}")
        return cb

    if progress:
        progress(0.0, "Scoring baseline parameters…")
    baseline = cv_rmse(name, y, base_params, cfg.folds, cfg.cv_test_size)  # raises if model itself is broken

    def make_objective(space):
        def objective(trial):
            params = {**base_params, **_suggest(trial, space)}
            try:
                return cv_rmse(name, y, params, cfg.folds, cfg.cv_test_size, trial if cfg.prune else None)
            except optuna.TrialPruned:
                raise
            except Exception as e:  # noqa: BLE001 - a failing config is pruned, not fatal
                trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                raise optuna.TrialPruned()
        return objective

    pruner = lambda: optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)  # noqa: E731
    study1 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed, multivariate=True),
                                 pruner=pruner())
    study1.optimize(make_objective(space1), n_trials=cfg.stage1_trials, timeout=max(cfg.timeout_s * 0.65, 5),
                    callbacks=[tick("Stage 1 · explore")])
    rows = _records(study1, 1, 0)
    df1 = pd.DataFrame(rows)
    complete1 = df1[df1["state"] == "COMPLETE"] if not df1.empty else df1

    space2 = space1
    if cfg.stage2_trials > 0 and len(complete1) >= 3:
        space2 = _refine(space1, complete1)
        study2 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed + 1),
                                     pruner=pruner())
        best1 = {k: study1.best_params[k] for k in space1}
        try:
            study2.enqueue_trial(best1)
        except Exception:  # noqa: BLE001
            pass
        remaining = max(cfg.timeout_s - (time.time() - t0), 5)
        done["n"] = len(study1.trials)
        study2.optimize(make_objective(space2), n_trials=cfg.stage2_trials, timeout=remaining,
                        callbacks=[tick("Stage 2 · refine")])
        rows += _records(study2, 2, len(study1.trials))

    trials = pd.DataFrame(rows)
    comp = trials[trials["state"] == "COMPLETE"]
    if comp.empty:
        raise RuntimeError("No Optuna trial completed — every configuration failed or was pruned. "
                           "Try more trials, a longer history, or different data.")
    best_row = comp.loc[comp["value"].idxmin()]
    best = {}
    for k, spec in space1.items():  # restore native python types so the values can drive sidebar widgets
        v = best_row[k]
        v = v.item() if hasattr(v, "item") else v
        best[k] = int(v) if spec[0] == "int" else next((c for c in spec[1] if c == v or str(c) == str(v)), v)

    try:  # importances need ≥2 completed trials with variation; optional
        imp = optuna.importance.get_param_importances(study1)
    except Exception:  # noqa: BLE001
        imp = {}

    if progress:
        progress(1.0, "Done")
    return TuneResult(
        model=name, best_params={**base_params, **best}, best_value=float(best_row["value"]),
        baseline_params=dict(base_params), baseline_value=float(baseline), trials=trials, importances=dict(imp),
        space1=space1, space2=space2, n_complete=int(len(comp)), n_pruned=int((trials["state"] == "PRUNED").sum()),
        seconds=time.time() - t0, cfg=cfg.__dict__.copy(),
    )
