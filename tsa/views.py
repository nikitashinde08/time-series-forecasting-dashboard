"""Tab renderers. Each takes an AppCtx and is wrapped in an error boundary by app.py."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import acf, pacf

from . import plots
from .config import COLORS, KEYMAP, MODEL_NAMES
from .cv import feasible, make_splits, run_cv, summarize
from .data import PrepInfo, fingerprint
from .diagnostics import detect_anomalies, run_adf, summary_table
from .models import run_model
from .report import build_chart_zip, build_html_dashboard, build_pdf
from .sidebar import Settings
from .tuning import TuneConfig, run_tuning
from .ui import callout, metric_cards, section, show_df, show_fig


@dataclass
class AppCtx:
    y: pd.Series
    prep: PrepInfo
    S: Settings
    test_n: int
    label: str
    source_label: str
    shared: dict = field(default_factory=dict)  # cross-tab results consumed by the Export tab


# =============================================================================== 1. Overview
def render_overview(c: AppCtx) -> None:
    y, prep = c.y, c.prep
    chg = (y.iloc[-1] / y.iloc[0] - 1) * 100 if y.iloc[0] else np.nan
    metric_cards([
        ("Observations", f"{len(y):,}", f"{y.index[0]:%d %b %Y} → {y.index[-1]:%d %b %Y}", COLORS["ARIMA"]),
        ("Latest value", f"{y.iloc[-1]:,.2f}", f"{chg:+.1f}% over window", COLORS["SARIMA"]),
        ("Mean / Std", f"{y.mean():,.1f}", f"σ = {y.std():,.2f}", COLORS["Prophet"]),
        ("Missing imputed", f"{prep.n_missing}", prep.method, COLORS["LSTM (MLP)"]),
        ("Frequency", prep.freq, f"{prep.n_duplicates} duplicate timestamps dropped", COLORS["muted"]),
    ])
    if prep.n_missing:
        callout("info", f"<b>{prep.n_missing}</b> missing timestamps/values were filled using <b>{prep.method}</b>.")
    else:
        callout("success", "No missing values detected after regularising the index.")
    show_fig(plots.fig_timeseries(y, st.toggle("Show moving average", True), c.S.roll_win), "ts_main", "Time series")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        section("Descriptive statistics")
        c.shared["summary"] = summary_table(y)
        show_df(c.shared["summary"])
    with c2:
        section("Stationarity — Augmented Dickey-Fuller", "H₀: the series has a unit root (non-stationary).")
        try:
            adf, adf_d = run_adf(y), run_adf(y.diff().dropna())
            c.shared["adf"] = adf
            ca, cb = st.columns(2)
            with ca:
                metric_cards([("ADF statistic", f"{adf['stat']:.3f}", f"lags used: {adf['lags']}", COLORS["ARIMA"])])
            with cb:
                metric_cards([("p-value", f"{adf['p']:.4f}", "threshold 0.05", COLORS["SARIMA"] if adf["stationary"] else COLORS["anomaly"])])
            show_df(pd.DataFrame({"Level": list(adf["crit"]), "Critical value": [round(v, 4) for v in adf["crit"].values()],
                                  "Statistic < critical?": ["Yes" if adf["stat"] < v else "No" for v in adf["crit"].values()]}), hide_index=True)
            if adf["stationary"]:
                callout("success", f"p = {adf['p']:.4f} &lt; 0.05 → <b>reject H₀</b>: series appears <b>stationary</b>.")
            else:
                callout("warn", f"p = {adf['p']:.4f} ≥ 0.05 → likely <b>non-stationary</b>. After first differencing p = {adf_d['p']:.4f} "
                                f"({'stationary — use d = 1' if adf_d['stationary'] else 'still non-stationary — consider d = 2'}).")
        except Exception as e:  # noqa: BLE001
            callout("error", f"ADF test failed: {e}")
    with st.expander("Preview processed data"):
        show_df(c.y.rename(c.label).to_frame().tail(200))


# =============================================================================== 2. Diagnostics
def render_diagnostics(c: AppCtx) -> None:
    y, S = c.y, c.S
    section("Trend & seasonal decomposition")
    try:
        dm = S.decomp_model
        if dm == "Multiplicative" and (y <= 0).any():
            callout("warn", "Multiplicative decomposition needs strictly positive data — using <b>Additive</b>.")
            dm = "Additive"
        if len(y) < 2 * S.decomp_period:
            raise ValueError(f"Need at least 2 × period ({2 * S.decomp_period}) observations.")
        show_fig(plots.fig_decomposition(y, dm, S.decomp_period), "decomp", "Decomposition")
    except Exception as e:  # noqa: BLE001
        callout("error", f"Decomposition failed: {e}")

    section("Anomaly detection", f"Method: {S.anom_method}. Tune thresholds in the sidebar.")
    mask, bands = detect_anomalies(y, S.anom_method, S.iqr_k, S.iqr_basis, S.roll_win, S.z_thr)
    show_fig(plots.fig_anomalies(y, mask, bands), "anom", "Anomalies")
    if mask.any():
        with st.expander(f"View {int(mask.sum())} flagged points"):
            show_df(y[mask].rename(c.label).to_frame())

    section("Distribution")
    d1, d2 = st.columns([3, 1])
    with d2:
        basis = st.radio("Distribution of", ["Levels", "Daily % change"])
        nb = st.slider("Bins", 10, 100, 40)
    ds = y if basis == "Levels" else y.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100
    with d1:
        show_fig(plots.fig_distribution(ds, nb, basis.lower()), "dist", "Distribution")

    section("Autocorrelation", "Bars outside the dashed 95% band indicate significant lags.")
    try:
        sa = y.diff().dropna() if st.checkbox("Use first difference", False) else y
        nl = int(min(S.lag_max, len(sa) // 2 - 1))
        conf = 1.96 / np.sqrt(len(sa))
        a1, a2 = st.columns(2)
        with a1:
            show_fig(plots.fig_corr(acf(sa, nlags=nl, fft=True), conf, "ACF", COLORS["ARIMA"]), "acf", "ACF")
        with a2:
            show_fig(plots.fig_corr(pacf(sa, nlags=nl, method="ywm"), conf, "PACF", COLORS["Prophet"]), "pacf", "PACF")
        callout("info", "PACF cut-off suggests AR order <i>p</i>; ACF cut-off suggests MA order <i>q</i>. Spikes at multiples of the season length hint at SARIMA terms.")
    except Exception as e:  # noqa: BLE001
        callout("error", f"ACF/PACF failed: {e}")


# =============================================================================== 3. Forecasting
def render_forecast(c: AppCtx) -> None:
    S, y, mc = c.S, c.y, c.S.model_choice
    section(f"{mc} forecast", f"Back-test on the last {c.test_n} observations, then refit on all data and forecast {S.horizon} steps ahead.")
    with st.spinner(f"Training {mc}…"):
        res = run_model(mc, y, S.horizon, c.test_n, S.freq, S.params[mc])
    if not res.ok:
        callout("error", f"<b>{mc} could not be fitted.</b><br><code>{res.error}</code>")
        if mc == "Prophet":
            callout("info", "Prophet needs a working <code>cmdstanpy</code> backend: <code>pip install --upgrade prophet cmdstanpy</code> then "
                            "<code>python -c \"import cmdstanpy; cmdstanpy.install_cmdstan()\"</code>. Other models are unaffected.")
        return
    m = res.metrics
    metric_cards([("RMSE", f"{m['RMSE']:,.3f}", "hold-out", COLORS[mc]), ("MAE", f"{m['MAE']:,.3f}", "hold-out", COLORS[mc]),
                  ("MAPE", f"{m['MAPE (%)']:.2f}%", "hold-out", COLORS[mc]), ("R²", f"{m['R²']:.3f}", "hold-out", COLORS[mc]),
                  ("Fit time", f"{res.seconds:.1f}s", "back-test + refit", COLORS["muted"])])
    hist = st.slider("History shown on chart", 30, len(y), min(len(y), 240), key="hist_pts")
    show_fig(plots.fig_forecast(y, res, c.test_n, hist), "fc_main", "Forecast")
    if m["R²"] < 0:
        callout("warn", "Negative R² means the model did worse than predicting the hold-out mean — try the Tuning tab.")
    if mc == "LSTM (MLP)":
        callout("info", "The neural model forecasts recursively; its interval is an approximation (residual σ·√h).")
    section("Forecast table")
    ft = pd.DataFrame({"Forecast": res.fut_pred, "Lower 95%": res.fut_lo, "Upper 95%": res.fut_hi}).round(3)
    ft.index.name = "Date"
    c.shared["forecast"], c.shared["forecast_model"] = ft, mc
    show_df(ft)
    st.download_button("⬇️ Download forecast CSV", ft.to_csv(), f"{mc.split()[0].lower()}_forecast.csv", "text/csv")


# =============================================================================== 4. Benchmark
def render_benchmark(c: AppCtx) -> None:
    S, y = c.S, c.y
    section("Model benchmark", f"All models back-tested on the same last {c.test_n} observations using the sidebar hyper-parameters.")
    with st.spinner("Benchmarking all models (cached after the first run)…"):
        results = [run_model(n, y, S.horizon, c.test_n, S.freq, S.params[n]) for n in MODEL_NAMES]
    good = [r for r in results if r.ok]
    for r in (r for r in results if not r.ok):
        callout("warn", f"<b>{r.name}</b> was excluded: <code>{r.error}</code>")
    if not good:
        callout("error", "No model could be fitted — adjust parameters or data.")
        return
    bench = pd.DataFrame([{"Model": r.name, **r.metrics, "Fit time (s)": round(r.seconds, 1)} for r in good]).sort_values("RMSE").reset_index(drop=True)
    bench.insert(0, "Rank", np.arange(1, len(bench) + 1))
    c.shared["bench"], c.shared["test_n"] = bench, c.test_n
    best = bench.iloc[0]
    st.markdown(f'<div class="best-banner">🏆 Top performer: <b>{best["Model"]}</b> — lowest RMSE (<b>{best["RMSE"]:,.3f}</b>), '
                f'MAPE {best["MAPE (%)"]:.2f}%, R² {best["R²"]:.3f}.</div>', unsafe_allow_html=True)
    hl = "background-color:#bbf7d0;color:#052e16;font-weight:700"
    styled = (bench.style.apply(lambda r: ["background-color:#ecfdf5;font-weight:700" if r["Rank"] == 1 else ""] * len(r), axis=1)
              .format({"RMSE": "{:,.3f}", "MAE": "{:,.3f}", "MAPE (%)": "{:.2f}", "R²": "{:.3f}", "Fit time (s)": "{:.1f}"})
              .highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], props=hl).highlight_max(subset=["R²"], props=hl))
    show_df(styled, hide_index=True)
    st.caption("Ranked by RMSE (lower is better). Green cells mark the best value per metric.")
    show_fig(plots.fig_compare(y, good, c.test_n), "cmp", "Hold-out comparison")
    cols = st.columns(4)
    for col, met in zip(cols, ["RMSE", "MAE", "MAPE (%)", "R²"]):
        with col:
            show_fig(plots.fig_metric_bar(bench, met), f"bar_{met}", "Benchmark RMSE" if met == "RMSE" else None)
    callout("info", "Metrics come from a <b>single chronological hold-out</b>; see the <b>Cross-Validation</b> tab for a more robust estimate.")


# =============================================================================== 5. Tuning (Optuna)
def _apply_params(model: str, params: dict) -> None:
    """on_click callback: push tuned values into the sidebar widgets (runs before the next script run)."""
    for k, key in KEYMAP[model].items():
        if k in params:
            st.session_state[key] = params[k]


def render_tuning(c: AppCtx) -> None:
    S, y, mc = c.S, c.y, c.S.model_choice
    base = S.params[mc]
    section(f"Hyper-parameter tuning · {mc}", "Multi-stage Optuna search: explore the full space, then narrow the ranges around the best trials and refine.")
    callout("info", "<b>How it works</b> — Baseline → <b>Stage 1</b> TPE exploration (median-pruned across CV folds) → <b>Stage 2</b> refined search in the narrowed space. "
                    "The objective is the mean RMSE over rolling-origin folds on the <i>training</i> data only; the hold-out used for benchmarking stays untouched.")
    sig = (mc, fingerprint(y), c.test_n)

    with st.form("tune_form"):
        g = st.columns(4)
        n1 = g[0].slider("Stage 1 trials", 5, 100, 20)
        n2 = g[1].slider("Stage 2 trials", 0, 60, 12)
        folds = g[2].slider("CV folds", 2, 6, 3)
        cv_ts = g[3].slider("Fold test window", 7, 90, int(min(c.test_n, 30)))
        g2 = st.columns(4)
        tmo = g2[0].slider("Time budget (s)", 20, 900, 90, 10)
        hist = g2[1].slider("History used (obs)", 200, 1500, 400, 50)
        seed = g2[2].number_input("Seed", 0, 9999, 42)
        prune = g2[3].toggle("Prune weak trials", True)
        go_btn = st.form_submit_button(f"🚀 Run Optuna tuning for {mc}", type="primary")

    if go_btn:
        bar = st.progress(0.0, text="Starting…")
        try:
            res = run_tuning(mc, y.iloc[:-c.test_n], base, TuneConfig(n1, n2, folds, cv_ts, tmo, hist, int(seed), prune),
                             progress=lambda f, t: bar.progress(f, text=t))
            st.session_state["tune"] = {"sig": sig, "res": res}
            bar.empty()
        except ImportError:
            bar.empty()
            callout("error", "Optuna is not installed. Run <code>pip install optuna</code>.")
        except Exception as e:  # noqa: BLE001
            bar.empty()
            callout("error", f"<b>Tuning failed.</b> <code>{type(e).__name__}: {e}</code>")

    stored = st.session_state.get("tune")
    if not stored:
        callout("info", "No tuning run yet. Configure the budget above and press <b>Run</b>.")
        return
    if stored["sig"] != sig:
        callout("warn", f"Stored tuning results are for <b>{stored['sig'][0]}</b> on different data/hold-out. Re-run to refresh.")
        return
    res = stored["res"]
    c.shared["tune"] = res
    imp = res.improvement_pct
    metric_cards([
        ("Best CV-RMSE", f"{res.best_value:.3f}", "tuned", COLORS["SARIMA"]),
        ("Baseline CV-RMSE", f"{res.baseline_value:.3f}", "sidebar parameters", COLORS["anomaly"]),
        ("Improvement", f"{imp:+.1f}%", "lower error is better", COLORS["SARIMA"] if imp > 0 else COLORS["Prophet"]),
        ("Trials", f"{res.n_complete}", f"{res.n_pruned} pruned/failed", COLORS["ARIMA"]),
        ("Duration", f"{res.seconds:.0f}s", f"stage-2 space narrowed", COLORS["muted"]),
    ])

    rows = []
    for k in res.space1:
        s1, s2 = res.space1[k], res.space2[k]
        fmt = (lambda s: f"{s[1]}–{s[2]}") if s1[0] == "int" else (lambda s: ", ".join(map(str, s[1])))
        rows.append({"Parameter": k, "Baseline": res.baseline_params.get(k), "Tuned": res.best_params.get(k),
                     "Stage-1 range": fmt(s1), "Stage-2 range": fmt(s2)})
    show_df(pd.DataFrame(rows).astype(str), hide_index=True)

    b1, b2 = st.columns([1, 3])
    b1.button("✅ Apply tuned parameters", on_click=_apply_params, args=(mc, res.best_params), type="primary",
              help="Copies the tuned values into the sidebar controls and re-runs every tab with them.")
    b2.caption("Applying updates the sidebar → forecast, benchmark and CV tabs all use the tuned values.")

    # ---- honest validation on the untouched hold-out
    section("Hold-out validation", "Tuned vs baseline parameters on the final hold-out that tuning never saw.")
    with st.spinner("Validating on hold-out…"):
        r0 = run_model(mc, y, S.horizon, c.test_n, S.freq, res.baseline_params)
        r1 = run_model(mc, y, S.horizon, c.test_n, S.freq, res.best_params)
    if r0.ok and r1.ok:
        t = pd.DataFrame([{"Config": "Baseline", **r0.metrics}, {"Config": "Tuned", **r1.metrics}])
        d = {k: r1.metrics[k] - r0.metrics[k] for k in ("RMSE", "MAE", "MAPE (%)", "R²")}
        t.loc[len(t)] = ["Δ (tuned − baseline)", *[d[k] for k in ("RMSE", "MAE", "MAPE (%)", "R²")]]
        show_df(t.style.format({k: "{:+.3f}" if False else "{:.3f}" for k in ("RMSE", "MAE", "MAPE (%)", "R²")}), hide_index=True)
        if r1.metrics["RMSE"] < r0.metrics["RMSE"]:
            callout("success", f"Tuning generalised: hold-out RMSE improved by <b>{-d['RMSE']:.3f}</b> "
                               f"({-d['RMSE'] / r0.metrics['RMSE'] * 100:.1f}%).")
        else:
            callout("warn", "Tuned parameters did <b>not</b> beat the baseline on the hold-out — likely CV over-fit or a regime change. "
                            "Try more folds, a longer history, or a wider time budget.")
    else:
        callout("warn", f"Validation skipped: {(r0.error or r1.error)}")

    section("Search diagnostics")
    show_fig(plots.fig_tune_history(res.trials, res.baseline_value), "tune_hist", "Tuning history")
    a, b = st.columns(2)
    with a:
        if res.importances:
            show_fig(plots.fig_tune_importance(res.importances), "tune_imp", "Tuning importance")
        else:
            callout("info", "Parameter importance needs ≥ 2 completed, varied trials.")
    with b:
        if res.n_complete >= 2:
            show_fig(plots.fig_tune_parallel(res.trials, list(res.space1)), "tune_par")
    with st.expander("All trials"):
        show_df(res.trials.sort_values("value"), hide_index=True)


# =============================================================================== 6. Cross-validation
def render_cv(c: AppCtx) -> None:
    S, y = c.S, c.y
    section("Time-series cross-validation", "Rolling-origin evaluation: the model is always trained on the past and tested on the future.")
    g = st.columns(5)
    model = g[0].selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(S.model_choice), key="cv_model")
    scheme = g[1].selectbox("Scheme", ["Expanding window", "Sliding window"], key="cv_scheme")
    n_sp = g[2].slider("Folds", 2, 10, 5, key="cv_folds")
    tsz = g[3].slider("Test window", 7, 90, int(min(S.horizon, 30)), key="cv_test")
    gap = g[4].slider("Gap", 0, 30, 0, key="cv_gap", help="Observations skipped between train and test (avoids leakage with lagged features).")
    window = len(y)
    if scheme == "Sliding window":
        window = st.slider("Training window (obs)", 60, max(61, len(y) - n_sp * tsz - gap), min(365, max(60, len(y) // 2)), key="cv_win")

    ok, why = feasible(len(y), n_sp, tsz, gap)
    if not ok:
        callout("error", why)
        return
    splits = make_splits(len(y), n_sp, tsz, gap, scheme, window)
    show_fig(plots.fig_cv_layout(splits, y, gap), "cv_layout", "CV split layout")

    params = S.params[model]
    tuned = st.session_state.get("tune")
    if tuned and tuned["res"].model == model:
        src = st.radio("Parameters", ["Sidebar parameters", "Tuned parameters (Optuna)"], horizontal=True, key="cv_src")
        if src.startswith("Tuned"):
            params = tuned["res"].best_params

    run = st.button(f"▶️ Run cross-validation for {model}", type="primary")
    cfg_key = (model, scheme, n_sp, tsz, gap, window, fingerprint(y), json.dumps(params, sort_keys=True, default=str))
    if run:
        st.session_state["cv_cfg"] = cfg_key
    if st.session_state.get("cv_cfg") != cfg_key:
        callout("info", "Adjust the split settings above, then press <b>Run cross-validation</b> to fit every fold (results are cached).")
        return

    with st.spinner(f"Fitting {len(splits)} folds…"):
        folds = run_cv(model, y, n_sp, tsz, gap, scheme, window, params)
    for f in (f for f in folds if not f.ok):
        callout("warn", f"Fold {f.spec.fold} failed: <code>{f.error}</code>")
    summ = summarize(folds, y)
    if summ.empty:
        callout("error", "Every fold failed — nothing to display.")
        return
    c.shared["cv_summary"] = summ
    c.shared["cv_meta"] = f"{model}, {scheme.lower()}, test window {tsz}" + (f", gap {gap}" if gap else "")

    r = summ["RMSE"]
    cv_pct = r.std(ddof=0) / r.mean() * 100
    metric_cards([("Mean RMSE", f"{r.mean():.3f}", f"± {r.std(ddof=0):.3f} across folds", COLORS[model]),
                  ("Mean MAE", f"{summ['MAE'].mean():.3f}", "", COLORS[model]), ("Mean MAPE", f"{summ['MAPE (%)'].mean():.2f}%", "", COLORS[model]),
                  ("Mean R²", f"{summ['R²'].mean():.3f}", "", COLORS[model]),
                  ("Stability", f"{cv_pct:.0f}%", "RMSE coeff. of variation", COLORS["SARIMA"] if cv_pct < 25 else COLORS["Prophet"])])
    if cv_pct >= 40:
        callout("warn", "RMSE varies strongly between folds — performance is regime-dependent; do not trust a single hold-out number.")
    show_fig(plots.fig_cv_forecasts(y, folds, model), "cv_fc", "CV predictions")
    a, b = st.columns(2)
    with a:
        metric = st.selectbox("Metric by fold", ["RMSE", "MAE", "MAPE (%)", "R²"], key="cv_metric")
        show_fig(plots.fig_cv_metrics(summ, metric), "cv_met", "CV fold stability" if metric == "RMSE" else None)
    with b:
        show_fig(plots.fig_cv_residuals(folds), "cv_res", "CV residuals")
    show_fig(plots.fig_cv_horizon(folds), "cv_hor", "CV error vs horizon")
    show_df(summ.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE (%)": "{:.2f}", "R²": "{:.3f}", "Time (s)": "{:.1f}"}), hide_index=True)


# =============================================================================== 7. Export
EXEC_DEFAULT = ["Time series", "Forecast", "Hold-out comparison", "Tuning history", "CV split layout", "CV fold stability", "CV predictions"]


def render_export(c: AppCtx) -> None:
    sh, S = c.shared, c.S
    figs: dict = st.session_state.get("_figs", {})
    section("Export center", "Executive PDF report, chart bundles and raw data — built from exactly what you see in the dashboard.")
    if not figs:
        callout("warn", "No charts available yet.")
        return
    title = st.text_input("Report title", "Time Series Analytics & Forecasting Report")
    names = list(figs)
    default = [n for n in EXEC_DEFAULT if n in figs] or names
    sel = st.multiselect("Charts to include", names, default=default)
    chosen = {n: figs[n] for n in sel}
    o = st.columns(4)
    inc_stats = o[0].checkbox("Descriptive stats", True)
    inc_fc = o[1].checkbox("Forecast table", True)
    inc_tune = o[2].checkbox("Tuning section", "tune" in sh)
    inc_cv = o[3].checkbox("CV section", "cv_summary" in sh)

    ctx = {"title": title, "source": c.source_label, "n_obs": len(c.y), "freq": c.prep.freq, "imputed": c.prep.n_missing,
           "date_range": f"{c.y.index[0]:%d %b %Y} to {c.y.index[-1]:%d %b %Y}", "adf": sh.get("adf"),
           "summary": sh.get("summary") if inc_stats else None, "bench": sh.get("bench"), "test_n": c.test_n,
           "model": sh.get("forecast_model"), "params": S.params.get(sh.get("forecast_model", ""), {}), "horizon": S.horizon,
           "forecast": sh.get("forecast") if inc_fc else None, "tune": sh.get("tune") if inc_tune else None,
           "cv_summary": sh.get("cv_summary") if inc_cv else None, "cv_meta": sh.get("cv_meta")}
    sig = (title, tuple(sel), inc_stats, inc_fc, inc_tune, inc_cv, fingerprint(c.y), str(S.params))

    def stash(kind, builder):
        try:
            with st.spinner(f"Building {kind}… (rendering charts can take a few seconds each)"):
                st.session_state[f"x_{kind}"] = {"sig": sig, **builder()}
        except ImportError as e:
            callout("error", f"Missing dependency for {kind}: <code>{e}</code>. Install with <code>pip install reportlab kaleido</code>.")
        except Exception as e:  # noqa: BLE001
            callout("error", f"{kind} export failed: <code>{type(e).__name__}: {e}</code>")

    def get(kind):
        v = st.session_state.get(f"x_{kind}")
        return v if v and v["sig"] == sig else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 📄 PDF report")
        st.caption("Executive summary, benchmark, charts, forecast, tuning & CV detail.")
        if st.button("Generate PDF", key="b_pdf", type="primary"):
            def _pdf():
                data, warn = build_pdf(ctx, chosen)
                return {"data": data, "warn": warn}
            stash("pdf", _pdf)
        if (r := get("pdf")):
            if r["warn"]:
                callout("warn", r["warn"])
            st.download_button("⬇️ Download PDF", r["data"], "ts_report.pdf", "application/pdf", key="d_pdf")
    with c2:
        st.markdown("##### 🖼️ Executive charts (ZIP)")
        st.caption("High-res white-background PNGs + interactive HTML per chart.")
        if st.button("Build chart bundle", key="b_zip"):
            def _zip():
                data, err = build_chart_zip(chosen)
                return {"data": data, "warn": f"PNG export unavailable ({err}); ZIP contains interactive HTML only." if err else ""}
            stash("zip", _zip)
        if (r := get("zip")):
            if r["warn"]:
                callout("warn", r["warn"])
            st.download_button("⬇️ Download ZIP", r["data"], "executive_charts.zip", "application/zip", key="d_zip")
    with c3:
        st.markdown("##### 🌐 Interactive dashboard (HTML)")
        st.caption("One light-themed page with every selected chart, fully interactive.")
        if st.button("Build HTML", key="b_html"):
            stash("html", lambda: {"data": build_html_dashboard(chosen, title), "warn": ""})
        if (r := get("html")):
            st.download_button("⬇️ Download HTML", r["data"], "executive_dashboard.html", "text/html", key="d_html")

    section("Raw data & configuration")
    d = st.columns(4)
    d[0].download_button("Processed series (CSV)", c.y.rename(c.label).to_csv(), "series.csv", "text/csv")
    if sh.get("forecast") is not None:
        d[1].download_button("Forecast (CSV)", sh["forecast"].to_csv(), "forecast.csv", "text/csv", key="d_fc")
    if sh.get("bench") is not None:
        d[2].download_button("Benchmark (CSV)", sh["bench"].to_csv(index=False), "benchmark.csv", "text/csv")
    cfg = {"model": S.model_choice, "params": S.params, "horizon": S.horizon, "freq": S.freq, "impute": S.impute,
           "tuned": sh["tune"].best_params if "tune" in sh else None}
    d[3].download_button("Configuration (JSON)", json.dumps(cfg, indent=2, default=str), "config.json", "application/json")
