"""Light executive styling: CSS injection + Plotly template."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

TEMPLATE = "exec_light"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#f8fafc; --panel:#ffffff; --border:#dbe3ee; --text:#0f172a; --muted:#475569;
  --accent:#0369a1; --good:#15803d; --warn:#b45309; --bad:#b91c1c;
}
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1400px; }
h1,h2,h3,h4 { color:#0f172a; letter-spacing:-.01em; }

/* Hero */
.hero { background:linear-gradient(120deg,#ffffff 0%,#e0f2fe 60%,#bae6fd 100%); border:1px solid #bfdbfe;
  border-radius:18px; padding:26px 32px; margin-bottom:18px; box-shadow:0 8px 24px rgba(15,23,42,.08); }
.hero h1 { margin:0; font-size:1.75rem; font-weight:800; color:#0c4a6e; }
.hero p { margin:6px 0 0; font-size:.95rem; color:#334155; }
.hero .pill { display:inline-block; margin:12px 8px 0 0; padding:3px 11px; font-size:.72rem; font-weight:600; letter-spacing:.05em;
  text-transform:uppercase; color:#075985; background:#fff; border:1px solid #7dd3fc; border-radius:999px; }

.hero .byline { display:inline-block; margin-bottom:10px; padding:4px 12px; font-size:.82rem; color:#0c4a6e;
  background:#fff; border:1px solid #7dd3fc; border-radius:999px; }

/* Metric cards */
.card-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:14px; margin:6px 0 18px; }
.metric-card { background:var(--panel); border:1px solid var(--border); border-left:5px solid var(--accent-c,#0284c7);
  border-radius:14px; padding:15px 18px; box-shadow:0 1px 3px rgba(15,23,42,.06); transition:transform .15s, box-shadow .15s; }
.metric-card:hover { transform:translateY(-2px); box-shadow:0 8px 20px rgba(15,23,42,.12); }
.metric-card .label { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); }
.metric-card .value { font-size:1.55rem; font-weight:700; color:#0f172a; margin-top:4px; line-height:1.15; }
.metric-card .sub { font-size:.76rem; color:var(--muted); margin-top:4px; }

/* Callouts */
.callout { border-radius:12px; padding:12px 16px; margin:10px 0; font-size:.9rem; line-height:1.5; border:1px solid; }
.callout.info    { background:#eff6ff; border-color:#93c5fd; color:#1e3a8a; }
.callout.success { background:#f0fdf4; border-color:#86efac; color:#14532d; }
.callout.warn    { background:#fffbeb; border-color:#fcd34d; color:#78350f; }
.callout.error   { background:#fef2f2; border-color:#fca5a5; color:#7f1d1d; }
.callout code { background:rgba(15,23,42,.08); padding:1px 6px; border-radius:6px; color:#0f172a; }

.section-title { font-size:1.1rem; font-weight:700; color:#0f172a; margin:22px 0 2px; }
.section-sub { font-size:.86rem; color:var(--muted); margin-bottom:10px; }

[data-testid="stPlotlyChart"] { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:6px;
  box-shadow:0 1px 3px rgba(15,23,42,.05); }
[data-testid="stDataFrame"]  { border:1px solid var(--border); border-radius:12px; overflow:hidden; }
[data-testid="stForm"] { background:var(--panel); border:1px solid var(--border); border-radius:14px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--border); flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { height:44px; padding:0 16px; border-radius:10px 10px 0 0; font-weight:600; color:var(--muted); }
.stTabs [aria-selected="true"] { background:#e0f2fe; color:#075985 !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--border); }
section[data-testid="stSidebar"] h2 { font-size:.9rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#0369a1; }

/* Best banner */
.best-banner { background:linear-gradient(90deg,#dcfce7,#f0fdf4); border:1px solid #86efac;
  border-radius:14px; padding:14px 18px; margin:8px 0 14px; color:#14532d; }
.best-banner b { color:#052e16; }

/* Buttons */
.stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"]>button {
  border-radius:10px; font-weight:600; border:1px solid #7dd3fc; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#0284c7; box-shadow:0 0 0 2px rgba(2,132,199,.18); }
</style>
"""


def register_plotly_template() -> None:
    """Create the 'exec_light' Plotly template: white canvas, soft grid, dark text for maximum legibility."""
    if TEMPLATE in pio.templates:
        return
    t = go.layout.Template(pio.templates["plotly_white"])
    grid = "#e2e8f0"
    t.layout.update(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1e293b", size=12),
        xaxis=dict(gridcolor=grid, zerolinecolor="#cbd5e1", linecolor="#94a3b8", tickfont=dict(color="#334155")),
        yaxis=dict(gridcolor=grid, zerolinecolor="#cbd5e1", linecolor="#94a3b8", tickfont=dict(color="#334155")),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#0f172a", bordercolor="#94a3b8"),
    )
    pio.templates[TEMPLATE] = t


def inject_styles() -> None:
    register_plotly_template()
    st.markdown(CSS, unsafe_allow_html=True)
