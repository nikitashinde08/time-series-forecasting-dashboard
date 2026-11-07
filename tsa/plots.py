"""All Plotly figure builders (light executive theme)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from statsmodels.tsa.seasonal import seasonal_decompose

from .config import COLORS
from .cv import FoldResult, SplitSpec
from .models import ModelResult
from .ui import style_fig

FOLD_PALETTE = ["#0284c7", "#16a34a", "#d97706", "#db2777", "#9333ea", "#ea580c", "#0d9488", "#65a30d", "#dc2626", "#4f46e5"]


def _rgba(hex_: str, a: float) -> str:
    r, g, b = (int(hex_[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


def _band(fig, x, lo, hi, color, name):
    fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=lo, line=dict(width=0), fill="tonexty", fillcolor=_rgba(color, 0.2),
                             name=name, hoverinfo="skip"))


# ------------------------------------------------------------------ EDA
def fig_timeseries(y: pd.Series, show_ma: bool, ma_win: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=y.index, y=y.values, name="Series", line=dict(color=COLORS["actual"], width=1.5)))
    if show_ma:
        fig.add_trace(go.Scatter(x=y.index, y=y.rolling(ma_win, min_periods=1).mean(), name=f"{ma_win}-pt MA",
                                 line=dict(color=COLORS["accent"], width=2, dash="dot")))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), rangeselector=dict(
        x=0, y=1.02, yanchor="bottom", bgcolor="#f1f5f9", activecolor="#bae6fd", font=dict(color="#0f172a"),
        buttons=[dict(count=1, label="1M", step="month", stepmode="backward"),
                 dict(count=3, label="3M", step="month", stepmode="backward"),
                 dict(count=6, label="6M", step="month", stepmode="backward"),
                 dict(count=1, label="1Y", step="year", stepmode="backward"), dict(step="all", label="All")]))
    fig = style_fig(fig, 520)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_decomposition(y: pd.Series, model: str, period: int) -> go.Figure:
    dec = seasonal_decompose(y, model=model.lower(), period=period, extrapolate_trend="freq")
    parts = [("Observed", dec.observed, COLORS["actual"]), ("Trend", dec.trend, COLORS["ARIMA"]),
             ("Seasonal", dec.seasonal, COLORS["SARIMA"]), ("Residual", dec.resid, COLORS["Prophet"])]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=[p[0] for p in parts])
    for i, (_, s, c) in enumerate(parts, start=1):
        if i == 4:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="markers", marker=dict(size=3, color=c), showlegend=False), row=i, col=1)
        else:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, line=dict(color=c, width=1.4), showlegend=False), row=i, col=1)
    return style_fig(fig, 700, f"{model} decomposition (period = {period})")


def fig_anomalies(y, mask, bands) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bands["upper"].index, y=bands["upper"].values, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bands["lower"].index, y=bands["lower"].values, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(100,116,139,0.18)", name="Normal band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=y.index, y=y.values, name="Series", line=dict(color=COLORS["actual"], width=1.3)))
    fig.add_trace(go.Scatter(x=y.index[mask], y=y[mask], mode="markers", name="Anomaly",
                             marker=dict(color=COLORS["anomaly"], size=9, symbol="circle-open", line=dict(width=2))))
    return style_fig(fig, 460, f"Anomalies detected: {int(mask.sum())}")


def fig_distribution(y: pd.Series, bins: int, label: str) -> go.Figure:
    v = y.dropna().values
    fig = go.Figure(go.Histogram(x=v, nbinsx=bins, histnorm="probability density", name="Histogram",
                                 marker=dict(color=_rgba(COLORS["ARIMA"], 0.5), line=dict(color="#ffffff", width=1))))
    xs = np.linspace(v.min(), v.max(), 300)
    fig.add_trace(go.Scatter(x=xs, y=gaussian_kde(v)(xs), name="KDE", line=dict(color=COLORS["Prophet"], width=2.5)))
    fig.update_layout(bargap=0.03, hovermode="closest")
    return style_fig(fig, 380, f"Distribution of {label}")


def fig_corr(values, conf, title, color) -> go.Figure:
    fig = go.Figure(go.Bar(x=np.arange(len(values)), y=values, marker_color=color, width=0.35))
    for s in (conf, -conf):
        fig.add_hline(y=s, line=dict(color=COLORS["muted"], dash="dash"))
    fig.update_xaxes(title="Lag")
    fig.update_layout(hovermode="closest", showlegend=False)
    return style_fig(fig, 340, title)


# ------------------------------------------------------------------ forecasting
def fig_forecast(y: pd.Series, res: ModelResult, test_n: int, history: int) -> go.Figure:
    c, hist = COLORS[res.name], y.iloc[-history:]
    fig = go.Figure(go.Scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color=COLORS["actual"], width=1.8)))
    _band(fig, res.test_pred.index, res.test_lo.values, res.test_hi.values, c, "95% CI (back-test)")
    fig.add_trace(go.Scatter(x=res.test_pred.index, y=res.test_pred.values, name="Predicted (hold-out)", line=dict(color=c, width=2.2)))
    _band(fig, res.fut_pred.index, res.fut_lo.values, res.fut_hi.values, c, "95% CI (future)")
    fig.add_trace(go.Scatter(x=res.fut_pred.index, y=res.fut_pred.values, name="Forecast", line=dict(color=c, width=2.4, dash="dash")))
    split = y.index[-test_n]
    fig.add_shape(type="line", x0=split, x1=split, yref="paper", y0=0, y1=1, line=dict(color=COLORS["muted"], dash="dot"))
    fig.add_annotation(x=split, yref="paper", y=1.0, text="train | test", showarrow=False, yanchor="bottom", font=dict(size=11, color="#475569"))
    return style_fig(fig, 540, f"{res.name}: actual vs predicted, with future forecast")


def fig_compare(y, results, test_n) -> go.Figure:
    ctx = y.iloc[-test_n * 3:]
    fig = go.Figure(go.Scatter(x=ctx.index, y=ctx.values, name="Actual", line=dict(color=COLORS["actual"], width=2)))
    for r in results:
        fig.add_trace(go.Scatter(x=r.test_pred.index, y=r.test_pred.values, name=r.name, line=dict(color=COLORS[r.name], width=2)))
    return style_fig(fig, 420, f"Hold-out comparison (last {test_n} observations)")


def fig_metric_bar(bench: pd.DataFrame, metric: str) -> go.Figure:
    f = go.Figure(go.Bar(x=bench["Model"], y=bench[metric], marker_color=[COLORS[m] for m in bench["Model"]],
                         text=bench[metric].round(3), textposition="outside"))
    f.update_layout(hovermode="closest", showlegend=False)
    return style_fig(f, 300, metric)


# ------------------------------------------------------------------ Optuna tuning
def fig_tune_history(trials: pd.DataFrame, baseline: float) -> go.Figure:
    comp = trials[trials["state"] == "COMPLETE"].sort_values("trial")
    fig = go.Figure()
    for stage, label in ((1, "Stage 1 · explore"), (2, "Stage 2 · refine")):
        d = comp[comp["stage"] == stage]
        fig.add_trace(go.Scatter(x=d["trial"], y=d["value"], mode="markers", name=label,
                                 marker=dict(size=9, color=COLORS[f"stage{stage}"], line=dict(width=1, color="#ffffff"))))
    fig.add_trace(go.Scatter(x=comp["trial"], y=comp["value"].cummin(), name="Best so far", mode="lines",
                             line=dict(color=COLORS["SARIMA"], width=2.5, shape="hv")))
    pr = trials[trials["state"] == "PRUNED"]
    if len(pr):
        fig.add_trace(go.Scatter(x=pr["trial"], y=[comp["value"].max()] * len(pr), mode="markers", name="Pruned / failed",
                                 marker=dict(symbol="x", size=8, color=COLORS["muted"]), hovertemplate="trial %{x} pruned<extra></extra>"))
    if np.isfinite(baseline):
        fig.add_hline(y=baseline, line=dict(color=COLORS["anomaly"], dash="dash"), annotation_text="baseline (current params)",
                      annotation_font_color=COLORS["anomaly"])
    fig.update_xaxes(title="Trial")
    fig.update_yaxes(title="CV RMSE (lower is better)")
    fig.update_layout(hovermode="closest")
    return style_fig(fig, 400, "Optimisation history")


def fig_tune_importance(imp: dict) -> go.Figure:
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=COLORS["accent"], text=[f"{v:.0%}" for _, v in items], textposition="outside"))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(range=[0, max(v for _, v in items) * 1.25])
    return style_fig(fig, 340, "Hyper-parameter importance")


def fig_tune_parallel(trials: pd.DataFrame, params: list[str]) -> go.Figure:
    d = trials[trials["state"] == "COMPLETE"]
    dims = []
    for p in params:
        col = d[p]
        if pd.api.types.is_numeric_dtype(col) and col.dtype != bool:
            dims.append(dict(label=p, values=col.astype(float)))
        else:
            cats = list(dict.fromkeys(col.astype(str)))
            dims.append(dict(label=p, values=col.astype(str).map({c: i for i, c in enumerate(cats)}),
                             tickvals=list(range(len(cats))), ticktext=cats))
    dims.append(dict(label="CV RMSE", values=d["value"]))
    fig = go.Figure(go.Parcoords(line=dict(color=d["value"], colorscale="Viridis_r", showscale=True), dimensions=dims))
    fig = style_fig(fig, 440, "Parallel coordinates (best trials = bright)")
    fig.update_layout(margin=dict(l=50, r=30, t=110, b=20))
    return fig
    return style_fig(fig, 420, "Parallel coordinates (best trials = bright)")


# ------------------------------------------------------------------ cross-validation
def fig_cv_layout(splits: list[SplitSpec], y: pd.Series, gap: int) -> go.Figure:
    """Gantt-style view of every split: train (blue), gap (grey), test (amber)."""
    n, fig = len(y), go.Figure()
    labels = [f"Fold {s.fold}" for s in splits]
    d = lambda i: f"{y.index[min(i, n - 1)]:%Y-%m-%d}"  # noqa: E731
    fig.add_trace(go.Bar(y=labels, x=[s.tr_end - s.tr_start for s in splits], base=[s.tr_start for s in splits], orientation="h",
                         name="Train", marker_color=_rgba(COLORS["accent"], 0.85),
                         customdata=[[d(s.tr_start), d(s.tr_end - 1), s.tr_end - s.tr_start] for s in splits],
                         hovertemplate="Train %{customdata[0]} → %{customdata[1]}<br>%{customdata[2]} obs<extra></extra>"))
    if gap:
        fig.add_trace(go.Bar(y=labels, x=[s.te_start - s.tr_end for s in splits], base=[s.tr_end for s in splits], orientation="h",
                             name="Gap", marker_color="rgba(100,116,139,.45)", hoverinfo="skip"))
    fig.add_trace(go.Bar(y=labels, x=[s.te_end - s.te_start for s in splits], base=[s.te_start for s in splits], orientation="h",
                         name="Test", marker_color=COLORS["stage2"],
                         customdata=[[d(s.te_start), d(s.te_end - 1), s.te_end - s.te_start] for s in splits],
                         hovertemplate="Test %{customdata[0]} → %{customdata[1]}<br>%{customdata[2]} obs<extra></extra>"))
    ticks = np.linspace(0, n - 1, 6).astype(int)
    fig.update_xaxes(tickvals=ticks, ticktext=[d(i) for i in ticks], range=[0, n])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="overlay", bargap=0.35, hovermode="closest")
    return style_fig(fig, 120 + 38 * len(splits), "Cross-validation split layout")


def fig_cv_forecasts(y: pd.Series, folds: list[FoldResult], model: str) -> go.Figure:
    good = [f for f in folds if f.ok]
    start = max(0, good[0].spec.te_start - len(good[0].actual) * 2) if good else 0
    ctx = y.iloc[start:]
    fig = go.Figure(go.Scatter(x=ctx.index, y=ctx.values, name="Actual", line=dict(color=COLORS["actual"], width=1.6)))
    for i, f in enumerate(good):
        c = FOLD_PALETTE[i % len(FOLD_PALETTE)]
        fig.add_vrect(x0=f.pred.index[0], x1=f.pred.index[-1], fillcolor=_rgba(c, 0.10), line_width=0)
        fig.add_trace(go.Scatter(x=f.pred.index, y=f.pred.values, name=f"Fold {f.spec.fold}", line=dict(color=c, width=2.2)))
    return style_fig(fig, 460, f"{model}: out-of-sample predictions per fold")


def fig_cv_metrics(summary: pd.DataFrame, metric: str = "RMSE") -> go.Figure:
    v = summary[metric]
    fig = go.Figure(go.Bar(x=[f"Fold {i}" for i in summary["Fold"]], y=v,
                           marker_color=[FOLD_PALETTE[i % len(FOLD_PALETTE)] for i in range(len(v))],
                           text=v.round(2), textposition="outside"))
    fig.add_hline(y=v.mean(), line=dict(color=COLORS["anomaly"], dash="dash"),
                  annotation_text=f"mean {v.mean():.2f} ± {v.std(ddof=0):.2f}", annotation_font_color=COLORS["anomaly"])
    fig.update_layout(hovermode="closest", showlegend=False)
    return style_fig(fig, 340, f"{metric} by fold (stability)")


def fig_cv_residuals(folds: list[FoldResult]) -> go.Figure:
    fig = go.Figure()
    for i, f in enumerate([f for f in folds if f.ok]):
        c = FOLD_PALETTE[i % len(FOLD_PALETTE)]
        fig.add_trace(go.Box(y=(f.actual - f.pred).values, name=f"Fold {f.spec.fold}", marker_color=c, boxmean=True))
    fig.add_hline(y=0, line=dict(color=COLORS["muted"], dash="dot"))
    fig.update_layout(hovermode="closest", showlegend=False)
    return style_fig(fig, 340, "Residual distribution by fold (actual − predicted)")


def fig_cv_horizon(folds: list[FoldResult]) -> go.Figure:
    good = [f for f in folds if f.ok]
    L = min(len(f.actual) for f in good)
    E = np.array([np.abs((f.actual - f.pred).values[:L]) for f in good])
    x = np.arange(1, L + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=E.max(0), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=E.min(0), line=dict(width=0), fill="tonexty", fillcolor=_rgba(COLORS["accent"], 0.2),
                             name="min–max across folds", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=E.mean(0), name="Mean |error|", line=dict(color=COLORS["accent"], width=2.5)))
    fig.update_xaxes(title="Steps ahead")
    fig.update_yaxes(title="Absolute error")
    return style_fig(fig, 340, "Error growth with forecast horizon")
