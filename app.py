"""TriMind AI — Home. Three Challenges. Three Predictions. One AI Platform."""

import base64
import os

import streamlit as st
import streamlit.components.v1 as components

from src.utils import (
    APP_NAME,
    SLOGAN,
    apply_custom_css,
    play_voice,
    render_footer,
    render_sidebar,
    show_robot,
)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Videos bigger than this are shown with st.video() instead of base64 embedding
MAX_EMBED_MB = 15


# ---------------- Video Player ----------------
def embed_video(
    path: str,
    muted: bool = True,
    height: int = 420,
    width: str = "100%",
):
    """
    Embed an MP4 video.

    The video starts muted where browser autoplay permits it.
    The user can click the custom button to enable sound and play.
    Large videos fall back to st.video() so the page stays fast.
    """

    if not os.path.exists(path):
        return False

    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > MAX_EMBED_MB:
            st.video(path)
            return True

        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        muted_value = "true" if muted else "false"
        autoplay_value = "autoplay" if muted else ""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{
                    box-sizing: border-box;
                }}

                body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: Arial, sans-serif;
                }}

                .video-container {{
                    width: {width};
                    text-align: center;
                }}

                video {{
                    display: block;
                    width: 100%;
                    max-height: {height}px;
                    border-radius: 12px;
                    outline: none;
                    background: #000;
                    object-fit: contain;
                }}

                .sound-button {{
                    margin-top: 12px;
                    padding: 12px 22px;
                    width: 100%;
                    border: none;
                    border-radius: 10px;
                    background: linear-gradient(135deg, #6541ff, #00bcd4);
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: opacity 0.2s ease;
                }}

                .sound-button:hover {{
                    opacity: 0.88;
                }}

                .status {{
                    margin-top: 8px;
                    color: #888;
                    font-size: 12px;
                }}
            </style>
        </head>

        <body>
            <div class="video-container">

                <video
                    id="introVideo"
                    {autoplay_value}
                    muted="{muted_value}"
                    playsinline
                    controls
                    preload="auto"
                >
                    <source
                        src="data:video/mp4;base64,{data}"
                        type="video/mp4"
                    >
                    Aapka browser video support nahi karta.
                </video>

                <button
                    id="soundButton"
                    class="sound-button"
                    type="button"
                >
                    🔊 Play Intro with Sound
                </button>

                <div id="status" class="status">
                    Sound ke saath video chalane ke liye button dabayein.
                </div>

            </div>

            <script>
                const video = document.getElementById("introVideo");
                const button = document.getElementById("soundButton");
                const status = document.getElementById("status");

                // Try muted autoplay first.
                video.muted = {str(muted).lower()};

                video.play().catch(function () {{
                    status.textContent =
                        "Video chalane ke liye Play button dabayein.";
                }});

                // Important: play directly inside the user's click event.
                button.addEventListener("click", async function () {{
                    try {{
                        video.muted = false;
                        video.volume = 1;

                        await video.play();

                        button.textContent = "🔊 Sound Enabled";
                        status.textContent =
                            "Intro video sound ke saath chal rahi hai.";
                    }} catch (error) {{
                        status.textContent =
                            "Video play nahi hui. Video par Play dabayein.";
                    }}
                }});

                video.addEventListener("error", function () {{
                    status.textContent =
                        "Video load nahi hui. assets/video/intro.mp4 check karein.";
                }});
            </script>
        </body>
        </html>
        """

        components.html(
            html,
            height=height + 90,
            scrolling=False,
        )

        return True

    except Exception as e:
        st.error(f"Video load error: {e}")
        return False


# ---------------- Welcome / Intro Screen ----------------

if "entered" not in st.session_state:
    st.session_state.entered = False

if "welcome_played" not in st.session_state:
    st.session_state.welcome_played = False

if not st.session_state.entered:

    # Intro screen par sidebar chhupa dein (saaf look)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">TriMind AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="hero-slogan">{SLOGAN}</div>',
        unsafe_allow_html=True,
    )

    if os.path.exists("assets/logo.png"):
        c1, c2, c3 = st.columns([1, 1, 1])

        with c2:
            st.image(
                "assets/logo.png",
                use_container_width=True,
            )

    intro = "assets/video/intro.mp4"

    if not embed_video(intro, muted=True):
        st.warning(
            "Intro video nahi mili. "
            "Check karein: assets/video/intro.mp4"
        )

    st.markdown("")

    b1, b2 = st.columns(2)

    with b1:
        if st.button(
            "Enter TriMind AI →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.entered = True
            st.session_state.play_welcome_now = True
            st.rerun()

    with b2:
        if st.button(
            "Skip →",
            use_container_width=True,
        ):
            st.session_state.entered = True
            st.session_state.play_welcome_now = False
            st.rerun()

    st.stop()


# ---------------- Sidebar ----------------

render_sidebar()


# ---------------- Welcome Voice ----------------

if (
    st.session_state.get("play_welcome_now", False)
    and not st.session_state.welcome_played
):
    play_voice("welcome")

    st.session_state.welcome_played = True
    st.session_state.play_welcome_now = False


# ---------------- Hero ----------------


@st.cache_data(show_spinner=False)
def _logo_b64(path: str, mtime: float) -> str:
    """Logo as base64 so it can sit inline next to the title (mtime busts cache)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


logo_html = ""
if os.path.exists("assets/logo.png"):
    logo_b64 = _logo_b64("assets/logo.png", os.path.getmtime("assets/logo.png"))
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" '
        'width="64" style="border-radius:14px;"/>'
    )

with st.container(key="hero_panel"):
    h_left, h_right = st.columns([3, 2], vertical_alignment="center")

    with h_left:
        st.markdown(
            f"""
            <div class="hero-badge">🚀 Unified AI Intelligence Platform</div>
            <div class="hero-head">{logo_html}<div class="hero-title left">TriMind AI</div></div>
            <div class="hero-slogan left">{SLOGAN}</div>
            <div class="hero-desc">
                Welcome to <b>TriMind AI</b> — a unified intelligence platform
                that turns raw business data into decisions. Pick a room below:
                analyse the past, predict the future, and download your own AI report.
            </div>
            <div class="chip-row">
                <span class="chip">📈 Sales Forecasting</span>
                <span class="chip">👥 Churn Prediction</span>
                <span class="chip">🧑‍💼 HR Attrition</span>
                <span class="chip">🤖 3 AI Models</span>
                <span class="chip">📄 Auto Reports</span>
            </div>
            <div class="hero-stats">
                <div class="hero-stat"><b>3</b><span>Intelligence Rooms</span></div>
                <div class="hero-stat"><b>3</b><span>Models per Room</span></div>
                <div class="hero-stat"><b>1-click</b><span>AI Report</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h_right:
        with st.container(key="hero_robot"):
            show_robot(height=260)


st.divider()


# ---------------- Three Intelligence Rooms ----------------

st.subheader("🧠 Choose Your Intelligence Room")

r1, r2, r3 = st.columns(3)


# Sales Analytics
with r1:
    st.markdown(
        """
        <div class="module-card">
            <div style="font-size:2rem">📈</div>
            <h3>Sales Analytics &amp; Profit Prediction</h3>
            <p>
                Forecast revenue trends and profitability with regression
                models — compare Linear Regression, Random Forest and
                Neural Networks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Enter Sales Room →",
        key="go_sales",
        use_container_width=True,
    ):
        st.switch_page("pages/1_Sales_Analytics.py")


# Customer Churn
with r2:
    st.markdown(
        """
        <div class="module-card">
            <div style="font-size:2rem">👥</div>
            <h3>Customer Churn Prediction</h3>
            <p>
                Identify at-risk customers before they leave — behaviour
                analysis, retention insights and churn probability scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Enter Churn Room →",
        key="go_churn",
        use_container_width=True,
    ):
        st.switch_page("pages/2_Customer_Churn.py")


# HR Analytics
with r3:
    st.markdown(
        """
        <div class="module-card">
            <div style="font-size:2rem">🧑‍💼</div>
            <h3>HR Analytics &amp; Employee Attrition</h3>
            <p>
                Predict which employees are likely to leave — workforce
                retention forecasts and satisfaction-driven insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Enter HR Room →",
        key="go_hr",
        use_container_width=True,
    ):
        st.switch_page("pages/3_HR_Analytics.py")


st.divider()


# ---------------- How It Works ----------------

st.subheader("⚙️ How It Works")

w1, w2, w3, w4 = st.columns(4)

steps = [
    (
        w1,
        "📥",
        "1. Load Data",
        "Use the built-in sample dataset or upload your own CSV.",
    ),
    (
        w2,
        "🧹",
        "2. Clean & Explore",
        "Automated cleaning plus interactive EDA charts.",
    ),
    (
        w3,
        "🤖",
        "3. Train AI Models",
        "Three models compete — the best one wins, fairly.",
    ),
    (
        w4,
        "🔮",
        "4. Predict & Report",
        "Live predictions plus a downloadable AI report.",
    ),
]

for col, emoji, title, description in steps:
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="font-size:1.8rem">{emoji}</div>
                <div class="kpi-value" style="font-size:1.05rem">
                    {title}
                </div>
                <div class="kpi-label">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------- Footer ----------------

render_footer()