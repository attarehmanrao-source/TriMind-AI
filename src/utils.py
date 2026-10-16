"""Shared helpers for TriMind AI: theme CSS, sidebar, footer, voice, reports."""
import base64
import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_NAME = "TriMind AI"
SLOGAN = "Three Challenges. Three Predictions. One AI Platform."
VERSION = "v1.0.0"

LOTTIE_ROBOT_URL = "https://assets10.lottiefiles.com/packages/lf20_tno6cg2w.json"


def apply_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

        /* ---------- Animations ---------- */
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes glowPulse {
            0%, 100% { box-shadow: 0 0 18px rgba(108, 99, 255, 0.18); }
            50%      { box-shadow: 0 0 34px rgba(139, 92, 246, 0.38); }
        }

        /* ---------- Base ---------- */
        /* Solid background on purpose: the Lottie robot iframe is painted with
           the theme colour #0E1117, so any gradient here makes it look like a
           pasted box. Keep this EXACTLY equal to config.toml backgroundColor. */
        .stApp {
            font-family: 'Poppins', sans-serif;
            background: #0E1117;
        }
        header[data-testid="stHeader"] { background: transparent; }
        footer { visibility: hidden; }
        .block-container { padding-top: 2rem; }
        hr { border-color: rgba(139, 92, 246, 0.25) !important; }

        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0E1117; }
        ::-webkit-scrollbar-thumb { background: #6C63FF; border-radius: 8px; }

        /* ---------- Hero ---------- */
        .hero-title {
            font-size: 3.4rem; font-weight: 800; text-align: center;
            background: linear-gradient(90deg, #a78bfa, #60a5fa, #f0abfc, #a78bfa);
            background-size: 300% 300%;
            animation: gradientShift 6s ease infinite;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem; letter-spacing: -1px;
        }
        .hero-slogan {
            text-align: center; color: #b8bce6; font-size: 1.15rem;
            margin-bottom: 1.5rem; letter-spacing: 0.3px;
        }

        /* ---------- Cards ---------- */
        .module-card {
            border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 18px;
            padding: 1.4rem 1.2rem; min-height: 240px;
            background: linear-gradient(145deg, rgba(48, 40, 98, 0.55), rgba(20, 22, 38, 0.85));
            backdrop-filter: blur(10px);
            animation: glowPulse 4s ease-in-out infinite;
            transition: transform 0.25s ease, border-color 0.25s ease;
            margin-bottom: 0.6rem;
        }
        .module-card:hover {
            transform: translateY(-6px);
            border-color: rgba(167, 139, 250, 0.9);
        }
        .module-card h3 { margin-top: 0.4rem; font-size: 1.2rem; color: #ffffff; }
        .module-card p { color: #b8bce6; font-size: 0.92rem; line-height: 1.55; }

        .kpi-card {
            border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 14px;
            padding: 0.9rem 1rem; text-align: center; min-height: 150px;
            background: linear-gradient(145deg, rgba(40, 36, 80, 0.55), rgba(26, 29, 41, 0.8));
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(108, 99, 255, 0.3);
        }
        .kpi-value { font-size: 1.7rem; font-weight: 700; color: #ffffff; }
        .kpi-label { color: #b8bce6; font-size: 0.85rem; }

        /* ---------- Dedication ---------- */
        .dedication {
            margin-top: 3rem; padding: 1.4rem; border-radius: 16px; text-align: center;
            border: 1px solid rgba(139, 92, 246, 0.35);
            background: linear-gradient(145deg, rgba(48, 40, 98, 0.4), rgba(26, 29, 41, 0.7));
        }
        .dedication h4 { margin-bottom: 0.4rem; }
        .dedication p { color: #c9cdef; font-size: 0.95rem; margin: 0.25rem 0; }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1A1D29 0%, #14122b 100%);
            border-right: 1px solid rgba(139, 92, 246, 0.25);
        }
        section[data-testid="stSidebar"] img { border-radius: 10px; }
        a[data-testid="stPageLink-NavLink"]:hover {
            background: rgba(139, 92, 246, 0.2); border-radius: 10px;
        }

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.6);
            background: linear-gradient(90deg, #6C63FF, #8b5cf6);
            color: #ffffff; font-weight: 600; transition: all 0.2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            box-shadow: 0 0 18px rgba(139, 92, 246, 0.6);
            transform: translateY(-2px);
            border-color: #a78bfa; color: #ffffff;
        }

        /* ---------- Widgets ---------- */
        div[data-testid="stMetric"] {
            border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 14px;
            padding: 0.7rem 1rem; background: rgba(26, 29, 41, 0.7);
        }
        div[data-testid="stFileUploader"] section {
            border: 1px dashed rgba(139, 92, 246, 0.55); border-radius: 14px;
            background: rgba(26, 29, 41, 0.5);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(139, 92, 246, 0.25);
            border-radius: 12px; overflow: hidden;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { color: #a78bfa; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #8b5cf6; }
        div[data-baseweb="select"] > div,
        .stTextInput input, .stNumberInput input {
            border-radius: 10px; border-color: rgba(139, 92, 246, 0.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        logo = "assets/logo.png"
        if os.path.exists(logo):
            st.image(logo, width=110)
        st.markdown(f"### {APP_NAME}")
        st.caption(SLOGAN)

        st.divider()
        st.markdown("**🎓 Student Developer**")
        card = "assets/card.png"
        if os.path.exists(card):
            st.image(card, use_container_width=True)
        else:
            st.info("Add your campus card as `assets/card.png`")
        st.markdown("**M ATTA UR REHMAN KHAN**")
        st.caption("AI-777760 · Artificial Intelligence and Data Science · SMIT")

        st.divider()
        st.page_link("app.py", label="🏠 Home")
        st.page_link("pages/1_Sales_Analytics.py", label="📈 Sales Analytics")
        st.page_link("pages/2_Customer_Churn.py", label="👥 Customer Churn")
        st.page_link("pages/3_HR_Analytics.py", label="🧑‍💼 HR Analytics")

        st.divider()
        st.caption(f"{APP_NAME} {VERSION}")


def render_footer():
    st.markdown(
        """
        <div class="dedication">
            <h4>🤲 Dedication</h4>
            <p><i>"Whatever I am today, I owe to my respected teacher,
            <b>Mr. Muhammad Azeem</b>, and my institution."</i></p>
            <p>May Allah bless my teacher with a long life, good health, honor, and success. Ameen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def play_voice(name: str, autoplay: bool = True):
    """
    Play an MP3 announcement from assets/audio/ without showing a player.

    Uses components.html (st.markdown does not run <script> tags).
    NOTE: Browsers block autoplay until the user interacts with the page
    at least once. After the first click, play_voice() will autoplay.
    """
    path = f"assets/audio/{name}.mp3"
    if not os.path.exists(path):
        return
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        auto = "autoplay" if autoplay else ""
        components.html(
            f"""
            <audio id="vp" {auto} style="display:none;" preload="auto">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <script>
                var a = document.getElementById('vp');
                if (a) {{ a.play().catch(function() {{ /* blocked by browser */ }}); }}
            </script>
            """,
            height=0,
        )
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def _robot_transparent_b64(path: str, mtime: float, tol: int = 18) -> str:
    """
    Remove the solid background of the robot image (flood-fill from the
    corners) so it blends into the page. `mtime` only busts the cache when
    the file changes.
    """
    from PIL import Image, ImageDraw

    img = Image.open(path).convert("RGBA")
    w, h = img.size
    for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        ImageDraw.floodfill(img, corner, (0, 0, 0, 0), thresh=tol)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def show_robot(height: int = 220):
    try:
        from streamlit_lottie import st_lottie
        import requests

        @st.cache_data(show_spinner=False)
        def _load(url):
            r = requests.get(url, timeout=8)
            return r.json() if r.status_code == 200 else None

        anim = _load(LOTTIE_ROBOT_URL)
        if anim:
            st_lottie(anim, height=height, key=f"robot_{height}")
            return
    except Exception:
        pass

    path = "assets/robot.png"
    if not os.path.exists(path):
        return
    try:
        b64 = _robot_transparent_b64(path, os.path.getmtime(path))
        st.markdown(
            f"""
            <style>
            @keyframes robotFloat {{
                0%, 100% {{ transform: translateY(0); }}
                50%      {{ transform: translateY(-8px); }}
            }}
            .robot-float {{
                display: block; margin: 0 auto; max-width: 100%;
                animation: robotFloat 4s ease-in-out infinite;
                filter: drop-shadow(0 0 18px rgba(139, 92, 246, 0.45));
            }}
            </style>
            <img class="robot-float" width="{min(height, 220)}"
                 src="data:image/png;base64,{b64}" alt="TriMind robot"/>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        # Pillow problem etc. -> plain image so the page never breaks
        st.image(path, width=min(height, 220))


def validate_upload(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if df.empty:
        return False, "The uploaded file has no rows."
    return True, "OK"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0E1117")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_html_report(title, intro, metrics: dict, figures_b64: list, notes: str) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in metrics.items()
    )
    imgs = "".join(
        f'<img src="data:image/png;base64,{b}" style="max-width:100%;margin:12px 0;border-radius:8px;"/>'
        for b in figures_b64
    )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title} — TriMind AI Report</title>
<style>body{{font-family:Arial,sans-serif;background:#0E1117;color:#eee;max-width:900px;margin:auto;padding:24px}}
h1{{color:#a78bfa}}table{{border-collapse:collapse;width:100%;margin:16px 0}}
td,th{{border:1px solid #444;padding:8px}}th{{background:#1A1D29}}</style>
</head><body>
<h1>🤖 {title}</h1><p><i>TriMind AI · {datetime.now().strftime('%Y-%m-%d %H:%M')}</i></p>
<p>{intro}</p>
<h2>📊 Model Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>
<h2>📈 Charts</h2>{imgs}
<h2>📝 Notes</h2><p>{notes}</p>
<hr><p><i>Dedicated to respected teacher Mr. Muhammad Azeem.</i></p>
</body></html>"""