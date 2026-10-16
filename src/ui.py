"""Shared UI helpers for TriMind AI room pages (header, KPI cards, chart themes)."""
import streamlit as st

PALETTE = ["#8b5cf6", "#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#fb923c"]


def _html(markup: str):
    """Indent hata kar render karta hai, taake Markdown HTML ko code block na bana de."""
    clean = "\n".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(clean, unsafe_allow_html=True)


def apply_page_css(accent_rgb: str = "139,92,246"):
    """Room pages ki CSS. accent_rgb se har room ka apna rang aata hai."""
    css = """
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
    background: var(--title-grad); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
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
    background: var(--c1); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
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
    text-align: center; padding: 1.5rem 1rem; border-radius: 20px; margin-top: 1rem;
    border: 1px solid var(--rc); background: var(--rbg);
    animation: fadeUp 0.5s ease both;
}
.res-card .rl { color: var(--rt); font-size: 0.95rem; font-weight: 600; }
.res-card .rv {
    font-size: 2.8rem; font-weight: 800; line-height: 1.2;
    background: var(--rg); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.res-card .rs { color: #fff; font-size: 1.05rem; font-weight: 600; margin-top: 0.2rem; }
.res-bar {
    height: 10px; border-radius: 999px; margin: 0.9rem auto 0; max-width: 420px;
    background: rgba(255,255,255,0.1); overflow: hidden;
}
.res-bar > div { height: 100%; border-radius: 999px; background: var(--rg); }

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
    _html(css.replace("__ACC__", accent_rgb))


def page_header(icon: str, title: str, subtitle: str, badges=(),
                title_grad: str = "linear-gradient(90deg,#6ee7b7,#60a5fa,#c4b5fd)"):
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


def section(num: int, name: str, desc: str = ""):
    _html(
        f"""
<div class="sec-head"><div class="sec-num">{num}</div><div class="sec-name">{name}</div></div>
<div class="sec-desc">{desc}</div>
"""
    )


def kpi(col, icon: str, value: str, label: str, grad: str):
    col.markdown(
        f'<div class="kpi2" style="--c1: {grad};"><div class="ico">{icon}</div>'
        f'<div class="val">{value}</div><div class="lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )


def best_banner(name: str, line: str):
    _html(
        f"""
<div class="best-banner"><div class="trophy">🏆</div>
<div><div class="bn">Best model: {name}</div>
<div class="bs">{line}</div></div></div>
"""
    )


def result_card(label: str, value: str, sub: str = "", pct: float = None,
                tone: str = "gold"):
    """tone: gold | risk | safe. pct (0-100) ho to neeche progress bar aati hai."""
    tones = {
        "gold": ("rgba(251,191,36,0.5)", "linear-gradient(145deg,rgba(251,191,36,0.14),rgba(244,114,182,0.1))",
                 "linear-gradient(90deg,#fbbf24,#f472b6)", "#FDE68A"),
        "risk": ("rgba(244,114,182,0.6)", "linear-gradient(145deg,rgba(244,114,182,0.18),rgba(239,68,68,0.1))",
                 "linear-gradient(90deg,#fb7185,#f472b6)", "#FBCFE8"),
        "safe": ("rgba(52,211,153,0.55)", "linear-gradient(145deg,rgba(52,211,153,0.16),rgba(96,165,250,0.1))",
                 "linear-gradient(90deg,#34d399,#60a5fa)", "#A7F3D0"),
    }
    rc, rbg, rg, rt = tones[tone]
    bar = f'<div class="res-bar"><div style="width:{min(max(pct, 0), 100):.1f}%"></div></div>' if pct is not None else ""
    sub_html = f'<div class="rs">{sub}</div>' if sub else ""
    _html(
        f"""
<div class="res-card" style="--rc:{rc}; --rbg:{rbg}; --rg:{rg}; --rt:{rt};">
<div class="rl">{label}</div><div class="rv">{value}</div>{sub_html}{bar}</div>
"""
    )


def style_fig(fig):
    """Plotly dark theme jo app se match kare."""
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


def style_mpl(fig, ax, grid: bool = True):
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#3b3f63")
    ax.tick_params(colors="white")
    if grid:
        ax.grid(axis="y", color="#8b5cf6", alpha=0.12)