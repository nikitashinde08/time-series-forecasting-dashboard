"""Reusable UI building blocks: callouts, metric cards, figure display + export registry."""
from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from .config import AUTHOR_FULL, AUTHOR_NAME, AUTHOR_TAGLINE, COLORS
from .styles import TEMPLATE


def callout(kind: str, text: str) -> None:
    """Coloured banner. kind: info | success | warn | error."""
    icons = {"info": "ℹ️", "success": "✅", "warn": "⚠️", "error": "⛔"}
    st.markdown(f'<div class="callout {kind}">{icons[kind]} {text}</div>', unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, str, str, str]]) -> None:
    """items = (label, value, sub, accent_hex)."""
    html = '<div class="card-grid">'
    for label, value, sub, accent in items:
        html += (f'<div class="metric-card" style="--accent-c:{accent}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, pills: list[str]) -> None:
    p = "".join(f'<span class="pill">{x}</span>' for x in pills)
    st.markdown(f'<div class="hero"><div class="byline">👩‍🎓 <b>{AUTHOR_NAME}</b> · {AUTHOR_TAGLINE}</div>'
                f'<h1>{title}</h1><p>{subtitle}</p>{p}</div>', unsafe_allow_html=True)



def style_fig(fig: go.Figure, height: int = 460, title: str = "") -> go.Figure:
    fig.update_layout(
        template=TEMPLATE, height=height, title=dict(text=title, x=0.01, font=dict(size=15)),
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ---- figure registry: every chart shown can be exported later (PDF / ZIP / HTML) ----------------
def reset_registry() -> None:
    st.session_state["_figs"] = {}


def register(name: str, fig: go.Figure) -> None:
    st.session_state.setdefault("_figs", {})[name] = fig


def show_fig(fig: go.Figure, key: str, name: Optional[str] = None) -> None:
    """Render full-width and (optionally) register for export under `name`."""
    if name:
        register(name, fig)
    cfg = {"displaylogo": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def show_df(df, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


__all__ = ["callout", "metric_cards", "section", "hero", "style_fig", "show_fig", "show_df",
           "register", "reset_registry", "COLORS"]
