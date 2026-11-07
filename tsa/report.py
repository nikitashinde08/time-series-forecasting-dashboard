"""Export engine: print-ready PDF report, PNG chart bundle (ZIP) and a standalone interactive HTML dashboard."""
from __future__ import annotations

import copy
import io
import zipfile
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from .config import AUTHOR_FULL, AUTHOR_NAME, PRINT_COLOR_MAP

NAVY, SLATE, ACCENT, GREEN = "#0b1220", "#334155", "#0ea5e9", "#16a34a"


# ------------------------------------------------------------------ figure → static image
def print_ready(fig: go.Figure) -> go.Figure:
    """Copy a figure and restyle it for white paper (fixes near-white traces and text)."""
    f = copy.deepcopy(fig)
    f.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#1f2937"))
    for tr in f.data:
        line = getattr(tr, "line", None)
        if line is not None and getattr(line, "color", None) in PRINT_COLOR_MAP:
            line.color = PRINT_COLOR_MAP[line.color]
        mk = getattr(tr, "marker", None)
        if mk is not None and isinstance(getattr(mk, "color", None), str) and mk.color in PRINT_COLOR_MAP:
            mk.color = PRINT_COLOR_MAP[mk.color]
    f.add_annotation(xref="paper", yref="paper", x=0.995, y=0.01, xanchor="right", yanchor="bottom", showarrow=False,
                     text=AUTHOR_FULL, font=dict(size=11, color="rgba(71,85,105,0.55)"))
    f.update_xaxes(rangeslider_visible=False, rangeselector_visible=False, gridcolor="#e5e7eb")
    f.update_yaxes(gridcolor="#e5e7eb")
    return f


def figs_to_png(figs: dict[str, go.Figure], width: int = 1200, scale: int = 2) -> tuple[dict[str, bytes], str]:
    """Render figures to PNG with Kaleido. Returns (name->bytes, error_message). Failures are non-fatal."""
    out: dict[str, bytes] = {}
    err = ""
    for name, fig in figs.items():
        try:
            f = print_ready(fig)
            out[name] = f.to_image(format="png", width=width, height=int(f.layout.height or 480) + 20, scale=scale)
        except Exception as e:  # noqa: BLE001
            err = (f"{type(e).__name__}: {e}")[:300]
            break  # if kaleido/Chrome is missing, every figure will fail - stop early
    return out, err


# ------------------------------------------------------------------ ZIP / HTML bundles
def build_chart_zip(figs: dict[str, go.Figure]) -> tuple[bytes, str]:
    """ZIP with a PNG per chart (when Kaleido works) plus interactive HTML for each chart."""
    pngs, err = figs_to_png(figs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, (name, fig) in enumerate(figs.items(), 1):
            slug = f"{i:02d}_{''.join(c if c.isalnum() else '_' for c in name).strip('_').lower()}"
            if name in pngs:
                z.writestr(f"png/{slug}.png", pngs[name])
            z.writestr(f"html/{slug}.html", fig.to_html(include_plotlyjs="cdn", full_html=True))
        if err:
            z.writestr("README.txt", f"PNG export unavailable ({err}).\nInstall Chrome for Kaleido: `plotly_get_chrome`.\n")
    return buf.getvalue(), err


def build_html_dashboard(figs: dict[str, go.Figure], title: str) -> bytes:
    """Single self-contained-ish HTML page (Plotly via CDN) containing every selected chart, light executive style."""
    parts = [f"<h2>{n}</h2>" + f.to_html(include_plotlyjs=False, full_html=False, config={"displaylogo": False})
             for n, f in figs.items()]
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            "<script src='https://cdn.plot.ly/plotly-3.0.1.min.js'></script>"
            "<style>body{background:#f8fafc;color:#0f172a;font-family:Inter,system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px}"
            "h1{margin:0 0 4px}h2{font-size:1.05rem;margin:28px 0 6px;color:#0369a1}.sub{color:#475569;margin-bottom:12px}"
            ".card{background:#fff;border:1px solid #dbe3ee;border-radius:14px;padding:8px;margin-bottom:8px}</style></head><body>"
            f"<h1>{title}</h1><div class='sub'>{AUTHOR_FULL} &nbsp;|&nbsp; Generated {datetime.now():%Y-%m-%d %H:%M}</div>"
            + "".join(f"<div class='card'>{p}</div>" for p in parts)
            + f"<div class='sub' style='text-align:center;margin-top:24px'>{AUTHOR_FULL}</div></body></html>")
    return html.encode("utf-8")


# ------------------------------------------------------------------ PDF
def _df_table(df: pd.DataFrame, col_widths=None, highlight_first: bool = False, font_size: int = 8):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    df = df.rename(columns=lambda c: str(c).replace("²", "2"))  # built-in PDF font lacks the superscript glyph
    data = [list(map(str, df.columns))] + df.astype(object).where(df.notna(), "").values.tolist()
    data = [[(f"{c:,.3f}" if isinstance(c, float) else str(c)) for c in row] for row in data]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), font_size),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if highlight_first:
        style += [("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dcfce7")), ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def build_pdf(ctx: dict, figs: dict[str, go.Figure]) -> tuple[bytes, str]:
    """
    Build the executive PDF. ctx keys (all optional except title): title, source, n_obs, date_range, freq, imputed,
    adf (dict), summary (DataFrame), model, params, horizon, test_n, bench (DataFrame), forecast (DataFrame),
    tune (TuneResult), cv_summary (DataFrame), cv_meta (str).  Returns (pdf_bytes, warning_text).
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=colors.HexColor(NAVY), spaceBefore=10, spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor(ACCENT), spaceBefore=8, spaceAfter=3)
    BODY = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor(SLATE))
    SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.5, textColor=colors.HexColor("#64748b"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=ctx.get("title", "Report"), author="Time Series Explorer")
    W = A4[0] - 32 * mm
    story, warn = [], ""

    # ---- cover banner
    banner = Table([[Paragraph(f"<font color='white' size=19><b>{ctx['title']}</b></font>", BODY)],
                    [Paragraph(f"<font color='#9fb3cf'>Prepared by {AUTHOR_NAME} &nbsp;|&nbsp; Source: {ctx.get('source', '-')} &nbsp;|&nbsp; Generated {datetime.now():%d %b %Y %H:%M}</font>", BODY)]],
                   colWidths=[W])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NAVY)), ("LEFTPADDING", (0, 0), (-1, -1), 14),
                                ("TOPPADDING", (0, 0), (0, 0), 16), ("BOTTOMPADDING", (0, -1), (-1, -1), 14)]))
    story += [banner, Spacer(1, 10)]

    # ---- executive summary
    story.append(Paragraph("Executive summary", H1))
    bullets = []
    if ctx.get("n_obs"):
        bullets.append(f"<b>Data:</b> {ctx['n_obs']:,} observations, {ctx.get('date_range', '')}, frequency {ctx.get('freq', '')}; "
                       f"{ctx.get('imputed', 0)} missing values imputed.")
    a = ctx.get("adf")
    if a:
        bullets.append(f"<b>Stationarity (ADF):</b> statistic {a['stat']:.3f}, p-value {a['p']:.4f} - series is "
                       f"<b>{'stationary' if a['stationary'] else 'non-stationary'}</b> at the 5% level.")
    b = ctx.get("bench")
    if b is not None and len(b):
        top = b.iloc[0]
        bullets.append(f"<b>Best model:</b> {top['Model']} (RMSE {top['RMSE']:.3f}, MAE {top['MAE']:.3f}, "
                       f"MAPE {top['MAPE (%)']:.2f}%, R2 {top['R²']:.3f}) on a {ctx.get('test_n', '?')}-point hold-out.")
    t = ctx.get("tune")
    if t is not None:
        bullets.append(f"<b>Tuning ({t.model}, Optuna):</b> CV-RMSE improved from {t.baseline_value:.3f} to {t.best_value:.3f} "
                       f"({t.improvement_pct:+.1f}%) over {t.n_complete} completed trials.")
    if ctx.get("cv_summary") is not None and len(ctx["cv_summary"]):
        s = ctx["cv_summary"]
        bullets.append(f"<b>Cross-validation:</b> mean RMSE {s['RMSE'].mean():.3f} (± {s['RMSE'].std(ddof=0):.3f}) across {len(s)} folds"
                       f"{' - ' + ctx['cv_meta'] if ctx.get('cv_meta') else ''}.")
    for x in bullets:
        story.append(Paragraph("• " + x, BODY))
    story.append(Spacer(1, 6))

    if b is not None and len(b):
        story += [Paragraph("Model benchmark", H2), _df_table(b, highlight_first=True), Spacer(1, 4),
                  Paragraph("Ranked by RMSE on a single chronological hold-out; lower RMSE/MAE/MAPE and higher R2 are better.", SMALL)]
    if ctx.get("summary") is not None:
        s = ctx["summary"].reset_index().rename(columns={"index": "Statistic"})
        story += [Paragraph("Descriptive statistics", H2), _df_table(s, col_widths=[W * 0.5, W * 0.5], font_size=8)]

    # ---- charts
    pngs, err = figs_to_png(figs) if figs else ({}, "")
    if err:
        warn = f"Charts could not be rendered ({err}). Install a Chrome for Kaleido with `plotly_get_chrome`."
        story.append(Paragraph("<i>Charts omitted: static image export (Kaleido/Chrome) is unavailable.</i>", SMALL))
    if pngs:
        story.append(PageBreak())
        story.append(Paragraph("Charts", H1))
        for name, data in pngs.items():
            fig = figs[name]
            h = (fig.layout.height or 480) / 1200 * W
            story.append(KeepTogether([Paragraph(name, H2), Image(io.BytesIO(data), width=W, height=min(h, 120 * mm)), Spacer(1, 6)]))

    # ---- forecast table
    fc = ctx.get("forecast")
    if fc is not None and len(fc):
        story += [PageBreak(), Paragraph(f"{ctx.get('model', '')} forecast ({ctx.get('horizon', len(fc))} steps)", H1)]
        if ctx.get("params"):
            story.append(Paragraph("<b>Parameters:</b> " + ", ".join(f"{k}={v}" for k, v in ctx["params"].items()), BODY))
        f2 = fc.copy()
        f2.insert(0, "Date", [d.strftime("%Y-%m-%d") for d in f2.index])
        story += [Spacer(1, 4), _df_table(f2.reset_index(drop=True), font_size=7.5)]

    # ---- tuning detail
    if t is not None:
        story += [PageBreak(), Paragraph("Hyper-parameter tuning (Optuna)", H1),
                  Paragraph(f"Two-stage TPE search with median pruning. Objective: mean RMSE over {t.cfg.get('folds')} rolling-origin folds "
                            f"(test window {t.cfg.get('cv_test_size')}), computed on the training portion only. "
                            f"Completed {t.n_complete}, pruned/failed {t.n_pruned}, {t.seconds:.0f}s.", BODY), Spacer(1, 4)]
        pm = pd.DataFrame({"Parameter": list(t.space1), "Baseline": [t.baseline_params.get(k) for k in t.space1],
                           "Tuned": [t.best_params.get(k) for k in t.space1]}).astype(str)
        story.append(_df_table(pm, col_widths=[W * 0.4, W * 0.3, W * 0.3], font_size=8.5))
    if ctx.get("cv_summary") is not None and len(ctx["cv_summary"]):
        story += [Paragraph("Cross-validation folds", H2), _df_table(ctx["cv_summary"].astype(str), font_size=7)]

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(16 * mm, 9 * mm, f"{ctx['title']}  |  {AUTHOR_FULL.replace(chr(183), chr(45))}  |  not financial advice")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue(), warn
