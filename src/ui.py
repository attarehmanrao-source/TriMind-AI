"""Shared UI helpers for TriMind AI room pages.

Provides the page header, section headings, KPI cards, best-model banner,
prediction result card and chart themes (Plotly / Matplotlib).

Public API (unchanged, so existing pages keep working):
    apply_page_css, page_header, section, kpi, best_banner,
    result_card, style_fig, style_mpl
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import streamlit as st

PALETTE = ["#8b5cf6", "#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#fb923c"]
DEFAULT_ACCENT_RGB = "139,92,246"
DEFAULT_TITLE_GRADIENT = "linear-gradient(90deg,#6ee7b7,#60a5fa,#c4b5fd)"


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _html(markup: str) -> None:
    """Render HTML through st.markdown.

    Every line is stripped so Markdown never treats indented HTML as a code block.
    """
    clean = "\n".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(clean, unsafe_allow_html=True)


def _clamp_pct(value: float) -> float:
    """Clamp a percentage to the 0-100 range; NaN/inf become 0."""
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0
    return min(max(float(value), 0.0), 100.0)


@dataclass(frozen=True)
class _Tone:
    """Colour set used by the result card."""
    border: str
    background: str
    gradient: str
    text: str


_TONES = {
    "gold": _Tone(
        border="rgba(251,191,36,0.5)",
        background="linear-gradient(145deg,rgba(251,191,36,0.14),rgba(244,114,182,0.1))",
        gradient="linear-gradient(90deg,#fbbf24,#f472b6)",
        text="#FDE68A",
    ),
    "risk": _Tone(
        border="rgba(244,114,182,0.6)",
        background="linear-gradient(145deg,rgba(244,114,182,0.18),rgba(239,68,68,0.1))",
        gradient="linear-gradient(90deg,#fb7185,#f472b6)",
        text="#FBCFE8",
    ),
    "safe": _Tone(
        border="rgba(52,211,153,0.55)",
        background="linear-gradient(145deg,rgba(52,211,153,0.16),rgba(96,165,250,0.1))",
        gradient="linear-gradient(90deg,#34d399,#60a5fa)",
        text="#A7F3D0",
    ),
}


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
_PAGE_CSS = """
<style>
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- Page header ---------- */
.pg-header {
    display: flex; align-items: center; gap: 1.1rem;
    padding: 1.4rem 1.6rem; border-radius: 22px;
    border: 1px solid rgba(139,92,246,0.32);
    background:
        radial-gradient(520px 200px at 10% 0%, rgba(__ACC__,0.18), transparent 70%),
        radial-gradient(520px 200px at 95% 100%, rgba(139,92,246,0.22), transparent 70%),
        #161A27;
    animation: fadeUp 0.6s ease both;
}
.pg-icon {
    width: 64px; height: 64px; border-radius: 18px; font-size: 2rem; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    background: rgba(__ACC__,0.16); border: 1px solid rgba(__ACC__,0.45);
}
.pg-title {
    font-size: 2rem; font-weight: 800; line-height: 1.15; margin: 0;
    color: #E9E5FF; /* fallback if gradient text is unsupported */
}
.pg-sub { color: #AEB3D9; font-size: 0.95rem; margin-top: 0.2rem; }
.pg-badges { display: flex; gap: 0.45rem; flex-wrap: wrap; margin-top: 0.6rem; }
.pg-badges span {
    padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    color: #c4b5fd; background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.4);
}

/* ---------- Section heads ---------- */
.sec-head { display: flex; align-items: center; gap: 0.7rem; margin: 2rem 0 0.2rem; }
.sec-num {
    width: 34px; height: 34px; border-radius: 50%; font-weight: 800; font-size: 0.9rem;
    color: #fff; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #6C63FF, #f472b6);
    box-shadow: 0 0 14px rgba(139,92,246,0.55);
}
.sec-name { font-size: 1.45rem; font-weight: 700; color: #fff; }
.sec-desc { color: #AEB3D9; font-size: 0.92rem; margin: 0 0 1rem 2.7rem; }

/* ---------- KPI cards ---------- */
.kpi2 {
    position: relative; overflow: hidden; text-align: center;
    padding: 1.1rem 0.8rem 1rem; border-radius: 18px; min-height: 130px;
    background: linear-gradient(145deg, rgba(48,40,98,0.5), rgba(22,26,39,0.92));
    border: 1px solid rgba(139,92,246,0.3); transition: all 0.25s ease;
    animation: fadeUp 0.7s ease both;
}
.kpi2::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--c1);
}
.kpi2:hover {
    transform: translateY(-5px); border-color: rgba(167,139,250,0.8);
    box-shadow: 0 10px 26px rgba(108,99,255,0.3);
}
.kpi2 .ico { font-size: 1.5rem; }
.kpi2 .val {
    font-size: 1.75rem; font-weight: 800; line-height: 1.2; margin: 0.15rem 0;
    color: #ffffff; /* fallback if gradient text is unsupported */
}
.kpi2 .lbl { color: #AEB3D9; font-size: 0.84rem; font-weight: 500; }

/* ---------- Charts ---------- */
div[data-testid="stPlotlyChart"], div[data-testid="stImage"] img { border-radius: 16px; }
div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(139,92,246,0.22); padding: 0.4rem;
    background: rgba(22,26,39,0.55);
}

/* ---------- Best model banner ---------- */
.best-banner {
    display: flex; align-items: center; gap: 1rem; padding: 1rem 1.3rem;
    border-radius: 18px; margin: 0.8rem 0 1rem;
    background: linear-gradient(120deg, rgba(52,211,153,0.18), rgba(96,165,250,0.12));
    border: 1px solid rgba(52,211,153,0.5);
}
.best-banner .trophy { font-size: 2.2rem; }
.best-banner .bn { color: #fff; font-size: 1.15rem; font-weight: 700; }
.best-banner .bs { color: #A7F3D0; font-size: 0.9rem; }

/* ---------- Prediction result ---------- */
.res-card {
    animation: fadeUp 0.5s ease both;
}
/* Result card colours and sizes are set inline in result_card(). */

/* ---------- Gradient text (progressive enhancement) ----------
   Applied ONLY when the browser supports background-clip:text, so the
   text can never turn transparent/invisible on unsupported browsers. */
@supports ((-webkit-background-clip: text) or (background-clip: text)) {
    .pg-title {
        background: var(--title-grad);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .rv-gold { background: linear-gradient(90deg,#fbbf24,#f472b6); }
    .rv-risk { background: linear-gradient(90deg,#fb7185,#f472b6); }
    .rv-safe { background: linear-gradient(90deg,#34d399,#60a5fa); }
    .rv-gold, .rv-risk, .rv-safe {
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi2 .val {
        background: var(--c1);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
}

div[data-testid="stForm"] {
    border: 1px solid rgba(139,92,246,0.3); border-radius: 18px;
    background: rgba(22,26,39,0.55); padding: 1.2rem;
}

@media (max-width: 768px) {
    .pg-title { font-size: 1.4rem; }
    .pg-header { padding: 1rem; }
}
</style>
"""


def apply_page_css(accent_rgb: str = DEFAULT_ACCENT_RGB) -> None:
    """Inject the room-page CSS. ``accent_rgb`` (e.g. "139,92,246") sets the room accent."""
    _html(_PAGE_CSS.replace("__ACC__", accent_rgb))


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def page_header(
    icon: str,
    title: str,
    subtitle: str,
    badges: Iterable[str] = (),
    title_grad: str = DEFAULT_TITLE_GRADIENT,
) -> None:
    """Render the page header with icon, title, subtitle and badge chips."""
    chips = "".join(f"<span>{b}</span>" for b in badges)
    _html(
        f"""
<div class="pg-header" style="--title-grad: {title_grad};">
<div class="pg-icon">{icon}</div>
<div>
<div class="pg-title">{title}</div>
<div class="pg-sub">{subtitle}</div>
<div class="pg-badges">{chips}</div>
</div>
</div>
"""
    )


def section(num: int, name: str, desc: str = "") -> None:
    """Render a numbered section heading with an optional description."""
    _html(
        f"""
<div class="sec-head"><div class="sec-num">{num}</div><div class="sec-name">{name}</div></div>
<div class="sec-desc">{desc}</div>
"""
    )


def kpi(col, icon: str, value: str, label: str, grad: str) -> None:
    """Render a KPI card inside the given Streamlit column."""
    col.markdown(
        f'<div class="kpi2" style="--c1: {grad};"><div class="ico">{icon}</div>'
        f'<div class="val">{value}</div><div class="lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )


def best_banner(name: str, line: str) -> None:
    """Render the highlighted 'best model' banner."""
    _html(
        f"""
<div class="best-banner"><div class="trophy">🏆</div>
<div><div class="bn">Best model: {name}</div>
<div class="bs">{line}</div></div></div>
"""
    )


_TONE_ICONS = {"gold": "💰", "risk": "⚠️", "safe": "✅"}


def result_card(
    label: str,
    value: str,
    sub: str = "",
    pct: Optional[float] = None,
    tone: str = "gold",
    icon: Optional[str] = None,
) -> None:
    """Render a prediction result card (e.g. predicted profit).

    Args:
        label: Small caption above the value.
        value: Main result text (already formatted, e.g. "Rs. 12,450").
        sub: Optional line below the value.
        pct: Optional 0-100 value; shows a progress bar when provided.
        tone: One of "gold", "risk" or "safe".
        icon: Emoji shown in the badge on top. Defaults to a tone-based icon
            (💰 / ⚠️ / ✅). Pass an empty string to hide it.
    """
    key = tone if tone in _TONES else "gold"
    t = _TONES[key]
    icon = _TONE_ICONS[key] if icon is None else icon

    # NOTE: colours are direct inline styles (no CSS variables) because
    # Streamlit's sanitiser can drop custom properties from inline styles.
    icon_html = (
        '<div style="width:56px; height:56px; margin:0 auto 0.6rem; border-radius:16px; '
        'display:flex; align-items:center; justify-content:center; font-size:1.9rem; '
        f'background:rgba(255,255,255,0.08); border:1px solid {t.border};">{icon}</div>'
        if icon
        else ""
    )
    bar = (
        '<div style="height:10px; border-radius:999px; margin:0.9rem auto 0; '
        'max-width:420px; background:rgba(255,255,255,0.1); overflow:hidden;">'
        f'<div style="height:100%; width:{_clamp_pct(pct):.1f}%; border-radius:999px; '
        f'background:{t.gradient};"></div></div>'
        if pct is not None
        else ""
    )
    sub_html = (
        '<div style="color:#ffffff; font-size:1.05rem; font-weight:600; '
        f'margin-top:0.2rem;">{sub}</div>'
        if sub
        else ""
    )
    _html(
        f"""
<div class="res-card" style="text-align:center; padding:1.5rem 1rem; border-radius:20px; margin-top:1rem; border:1px solid {t.border}; background:{t.background};">
{icon_html}
<div style="color:{t.text}; font-size:0.95rem; font-weight:600;">{label}</div>
<div class="rv-{key}" style="color:{t.text}; font-size:2.8rem; font-weight:800; line-height:1.2;">{value}</div>
{sub_html}
{bar}
</div>
"""
    )


# --------------------------------------------------------------------------- #
# Chart themes
# --------------------------------------------------------------------------- #
def style_fig(fig):
    """Apply the app's dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", color="#E6E8FF"),
        title_font=dict(size=16, color="#ffffff"),
        margin=dict(l=20, r=20, t=55, b=20),
        colorway=PALETTE,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(139,92,246,0.12)")
    fig.update_yaxes(gridcolor="rgba(139,92,246,0.12)")
    return fig


def style_mpl(fig, ax, grid: bool = True) -> None:
    """Apply the app's dark theme to a Matplotlib figure/axes."""
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3b3f63")
    ax.tick_params(colors="white")
    if grid:
        ax.grid(axis="y", color="#8b5cf6", alpha=0.12)