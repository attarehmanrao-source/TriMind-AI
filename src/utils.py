#src/utils.py
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

APP_NAME = "TriMind AI"
SLOGAN = "Three Challenges. Three Predictions. One AI Platform."
VERSION = "v1.0.0"

LOTTIE_ROBOT_URL = "https://assets10.lottiefiles.com/packages/lf20_tno6cg2w.json"


def apply_custom_css():
    st.markdown(
        """
        <style>
        .hero-title {
            font-size: 3.2rem; font-weight: 800; text-align: center;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .hero-slogan { text-align: center; color: #b8bce6; font-size: 1.15rem; margin-bottom: 1.5rem; }
        .module-card {
            border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 16px;
            padding: 1.4rem 1.2rem; background: rgba(26, 29, 41, 0.75);
            box-shadow: 0 0 24px rgba(108, 99, 255, 0.12); height: 100%;
        }
        .module-card h3 { margin-top: 0.4rem; font-size: 1.25rem; }
        .module-card p { color: #b8bce6; font-size: 0.92rem; }
        .kpi-card {
            border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px;
            padding: 0.9rem 1rem; background: rgba(26, 29, 41, 0.7); text-align: center;
        }
        .kpi-value { font-size: 1.7rem; font-weight: 700; }
        .kpi-label { color: #b8bce6; font-size: 0.85rem; }
        .dedication {
            margin-top: 3rem; padding: 1.4rem; border-radius: 14px; text-align: center;
            border: 1px solid rgba(139, 92, 246, 0.35); background: rgba(26, 29, 41, 0.6);
        }
        .dedication h4 { margin-bottom: 0.4rem; }
        .dedication p { color: #c9cdeF; font-size: 0.95rem; margin: 0.25rem 0; }
        section[data-testid="stSidebar"] img { border-radius: 10px; }
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
            st.image(card, use_column_width=True)
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
    Play an MP3 announcement from assets/audio/ WITHOUT showing a player.

    Uses HTML5 <audio autoplay> with base64-embedded MP3 so the sound
    starts on its own (no play button visible to the user).

    NOTE: Browsers block autoplay until the user interacts with the page
    at least once (Chrome/Edge/Firefox policy). After the first click
    anywhere in the app, all subsequent play_voice() calls will autoplay.
    """
    path = f"assets/audio/{name}.mp3"
    if not os.path.exists(path):
        return
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        autoplay_attr = "autoplay" if autoplay else ""
        audio_html = f"""
        <audio {autoplay_attr} style="display:none;" preload="auto">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        <script>
        // Try to force-play; ignore errors if browser blocks autoplay.
        (function() {{
            var audios = window.parent.document.querySelectorAll('audio');
            if (audios.length) {{
                var a = audios[audios.length - 1];
                a.play().catch(function() {{ /* blocked by browser, silent */ }});
            }}
        }})();
        </script>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception:
        pass


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
    if os.path.exists("assets/robot.png"):
        st.image("assets/robot.png", width=min(height, 220))


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


