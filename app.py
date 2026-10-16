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


def _html(markup: str):
    """Indent hata kar render karta hai, taake Markdown HTML ko code block na bana de."""
    clean = "\n".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(clean, unsafe_allow_html=True)


# ---------------- Home page styling ----------------
def apply_home_css():
    _html(
        """
<style>
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes floatY {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-6px); }
}

/* ---------- Hero panel (pure black) ---------- */
.st-key-hero_panel {
    position: relative; overflow: hidden;
    padding: 2rem 2rem; border-radius: 26px;
    border: 1px solid rgba(139, 92, 246, 0.32);
    background:
        radial-gradient(620px 240px at 12% 0%, rgba(139, 92, 246, 0.18), transparent 70%),
        radial-gradient(520px 240px at 95% 100%, rgba(96, 165, 250, 0.12), transparent 70%),
        #000000;
    animation: fadeUp 0.7s ease both;
}
.hero-badge {
    display: inline-block; padding: 0.3rem 0.95rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; color: #d8ccff;
    background: rgba(139, 92, 246, 0.16);
    border: 1px solid rgba(139, 92, 246, 0.4); margin-bottom: 0.9rem;
}
.hero-head { display: flex; align-items: center; gap: 0.9rem; }
.hero-title.left {
    text-align: left; font-size: 3.4rem; line-height: 1.05;
    margin: 0; letter-spacing: -1.5px;
}
.hero-slogan.left {
    text-align: left; color: #c4c8f2; font-size: 1.08rem;
    font-weight: 500; margin: 0.5rem 0 0.9rem 0;
}
.hero-desc { color: #C9CEF2; font-size: 1rem; line-height: 1.85; margin-bottom: 1rem; }
.hero-desc b.brand {
    background-image: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.hl-past   { color: #60a5fa; font-weight: 600; }
.hl-future { color: #f0abfc; font-weight: 600; }
.hl-report { color: #34d399; font-weight: 600; }

/* ---------- Colored chips ---------- */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-bottom: 1.3rem; }
.chip {
    padding: 0.42rem 0.95rem; border-radius: 999px; font-size: 0.85rem;
    font-weight: 600; border: 1px solid transparent;
    transition: all 0.2s ease; cursor: default;
}
.chip:hover { transform: translateY(-3px) scale(1.04); }
.chip.c-sales  { background: rgba(52,211,153,0.15);  border-color: rgba(52,211,153,0.55);  color: #6ee7b7; }
.chip.c-churn  { background: rgba(96,165,250,0.15);  border-color: rgba(96,165,250,0.55);  color: #93c5fd; }
.chip.c-hr     { background: rgba(251,191,36,0.15);  border-color: rgba(251,191,36,0.55);  color: #fcd34d; }
.chip.c-models { background: rgba(167,139,250,0.18); border-color: rgba(167,139,250,0.6);  color: #c4b5fd; }
.chip.c-report { background: rgba(244,114,182,0.15); border-color: rgba(244,114,182,0.55); color: #f9a8d4; }
.chip.c-sales:hover  { box-shadow: 0 0 16px rgba(52,211,153,0.5); }
.chip.c-churn:hover  { box-shadow: 0 0 16px rgba(96,165,250,0.5); }
.chip.c-hr:hover     { box-shadow: 0 0 16px rgba(251,191,36,0.5); }
.chip.c-models:hover { box-shadow: 0 0 16px rgba(167,139,250,0.55); }
.chip.c-report:hover { box-shadow: 0 0 16px rgba(244,114,182,0.5); }

/* ---------- Hero stats ---------- */
.hero-stats {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem;
}
.hero-stat {
    position: relative; overflow: hidden; text-align: center;
    padding: 1.1rem 0.6rem 0.9rem; border-radius: 16px;
    background: linear-gradient(145deg, rgba(48,40,98,0.5), rgba(10,10,18,0.95));
    border: 1px solid rgba(139,92,246,0.3); transition: all 0.25s ease;
}
.hero-stat .bar {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.hero-stat:hover {
    transform: translateY(-5px); border-color: rgba(167,139,250,0.8);
    box-shadow: 0 10px 26px rgba(108,99,255,0.3);
}
.hero-stat .ico { display: block; font-size: 1.5rem; margin-bottom: 0.1rem; }
.hero-stat b {
    display: block; font-size: 1.9rem; font-weight: 800; line-height: 1.15;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-stat span.lbl { display: block; color: #AEB3D9; font-size: 0.82rem; margin-top: 0.15rem; }

/* Robot: black background hero mein ghul jaye */
.st-key-hero_robot {
    animation: floatY 5s ease-in-out infinite;
    mix-blend-mode: screen;
}
.st-key-hero_robot iframe,
.st-key-hero_robot img { background: #000000; }

/* ---------- Section titles ---------- */
.sec-title { margin: 1.6rem 0 0.15rem; font-size: 1.55rem; font-weight: 700; color: #fff; }
.sec-title span {
    background-image: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sec-sub { color: #AEB3D9; font-size: 0.95rem; margin-bottom: 1.1rem; }

/* ---------- Room cards ---------- */
.room-card {
    position: relative; overflow: hidden; min-height: 290px;
    padding: 1.5rem 1.3rem 1.2rem; border-radius: 22px; margin-bottom: 0.7rem;
    background: linear-gradient(145deg, rgba(48,40,98,0.55), rgba(20,22,38,0.92));
    border: 1px solid rgba(139,92,246,0.3);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    animation: fadeUp 0.8s ease both;
}
.room-card .bar {
    position: absolute; top: 0; left: 0; right: 0; height: 4px;
}
.room-card:hover { transform: translateY(-7px); }
.room-icon {
    width: 58px; height: 58px; border-radius: 16px; display: flex;
    align-items: center; justify-content: center; font-size: 1.9rem;
    margin-bottom: 0.8rem;
}
.room-card h3 { margin: 0.2rem 0 0.5rem; font-size: 1.15rem; color: #fff; line-height: 1.35; }
.room-card p  { color: #B6BBE0; font-size: 0.9rem; line-height: 1.65; margin-bottom: 0.9rem; }
.room-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.room-tags span {
    padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.74rem;
    font-weight: 600; background: rgba(255,255,255,0.03);
}

/* Sales (green) */
.room-card.sales .bar { background-image: linear-gradient(90deg,#34d399,#60a5fa); }
.room-card.sales .room-icon { background: rgba(52,211,153,0.22); }
.room-card.sales .room-tags span { color: #6ee7b7; border: 1px solid #6ee7b7; }
.room-card.sales:hover { border-color: #6ee7b7;
    box-shadow: 0 14px 34px rgba(0,0,0,0.35), 0 0 22px rgba(52,211,153,0.28); }

/* Churn (blue) */
.room-card.churn .bar { background-image: linear-gradient(90deg,#60a5fa,#a78bfa); }
.room-card.churn .room-icon { background: rgba(96,165,250,0.22); }
.room-card.churn .room-tags span { color: #93c5fd; border: 1px solid #93c5fd; }
.room-card.churn:hover { border-color: #93c5fd;
    box-shadow: 0 14px 34px rgba(0,0,0,0.35), 0 0 22px rgba(96,165,250,0.28); }

/* HR (yellow) */
.room-card.hr .bar { background-image: linear-gradient(90deg,#fbbf24,#f472b6); }
.room-card.hr .room-icon { background: rgba(251,191,36,0.22); }
.room-card.hr .room-tags span { color: #fcd34d; border: 1px solid #fcd34d; }
.room-card.hr:hover { border-color: #fcd34d;
    box-shadow: 0 14px 34px rgba(0,0,0,0.35), 0 0 22px rgba(251,191,36,0.28); }

/* ---------- How it works ---------- */
.step-card {
    position: relative; text-align: center; min-height: 190px;
    padding: 1.5rem 1rem 1rem; border-radius: 18px;
    background: linear-gradient(145deg, rgba(40,36,80,0.55), rgba(26,29,41,0.85));
    border: 1px solid rgba(139,92,246,0.28); transition: all 0.25s ease;
    animation: fadeUp 0.9s ease both;
}
.step-card:hover {
    transform: translateY(-5px); border-color: rgba(167,139,250,0.8);
    box-shadow: 0 10px 26px rgba(108,99,255,0.3);
}
.step-num {
    position: absolute; top: -14px; left: 50%; transform: translateX(-50%);
    width: 30px; height: 30px; border-radius: 50%; font-size: 0.85rem;
    font-weight: 800; color: #fff; display: flex; align-items: center;
    justify-content: center;
    background-image: linear-gradient(135deg, #6C63FF, #f472b6);
    box-shadow: 0 0 14px rgba(139,92,246,0.6);
}
.step-emoji { font-size: 2rem; margin: 0.3rem 0 0.35rem; }
.step-title { color: #fff; font-weight: 700; font-size: 1.02rem; margin-bottom: 0.35rem; }
.step-desc  { color: #AEB3D9; font-size: 0.86rem; line-height: 1.55; }

/* ---------- Mobile ---------- */
@media (max-width: 768px) {
    .hero-title.left { font-size: 2.3rem; }
    .st-key-hero_panel { padding: 1.3rem 1rem; }
    .hero-stats { grid-template-columns: 1fr; }
}
</style>
"""
    )


def section_head(title: str, sub: str = ""):
    _html(
        f"""
<div class="sec-title"><span>{title}</span></div>
<div class="sec-sub">{sub}</div>
"""
    )


# ---------------- Neuron / ML / Deep Learning panel ----------------
ML_PANEL_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent; overflow: hidden;
               font-family: 'Poppins', Arial, sans-serif; }
  .stage { position: relative; width: 100%; height: 215px; }
  .stage > * { position: absolute; }

  @keyframes floatY   { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-9px); } }
  @keyframes floatBig { 0%,100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-9px) rotate(7deg); } }
  @keyframes wobble   { 0%,100% { transform: rotate(-7deg); } 50% { transform: rotate(7deg); } }
  @keyframes dashMove { to { stroke-dashoffset: -20; } }
  @keyframes nodePulse{ 0%,100% { opacity: .45; } 50% { opacity: 1; } }
  @keyframes twinkle  { 0%,100% { opacity: .5; } 50% { opacity: 1; } }

  /* neural network */
  .net { left: 20%; top: 8px; width: 60%; }
  .net .edges line { stroke: rgba(167,139,250,.45); stroke-width: 1.3;
                     stroke-dasharray: 5 5; animation: dashMove 2.5s linear infinite; }
  .net circle { fill: #8b5cf6; stroke: #c4b5fd; stroke-width: 1.5;
                animation: nodePulse 2.4s ease-in-out infinite; }
  .net circle.out { fill: #34d399; stroke: #a7f3d0; }
  .net text { fill: #8f94c9; font-size: 9px; }

  /* neuron characters */
  .buddy { filter: drop-shadow(0 0 10px rgba(167,139,250,.6)); }
  .b1 { left: 1%;  top: 14px;    width: 76px; animation: floatY 4.5s ease-in-out infinite; }
  .b2 { right: 1%; bottom: 8px;  width: 84px; animation: wobble 3.6s ease-in-out infinite; }

  /* floating emojis */
  .emo { font-size: 1.7rem; filter: drop-shadow(0 0 10px rgba(139,92,246,.65));
         animation: floatBig 5s ease-in-out infinite; }
  .e1 { left: 21%; bottom: 6px;  animation-delay: .1s; }
  .e2 { left: 33%; bottom: 40px; animation-delay: .9s; }
  .e3 { left: 45%; bottom: 4px;  animation-delay: 1.5s; }
  .e4 { left: 57%; bottom: 42px; animation-delay: .5s; }
  .e5 { left: 69%; bottom: 6px;  animation-delay: 1.2s; }
  .e6 { right: 3%; top: 10px;    animation-delay: .7s; }
  .e7 { left: 9%;  bottom: 30px; animation-delay: 1.8s; }

  /* ML symbols */
  .sym { font-family: 'Courier New', monospace; font-weight: 700; font-size: .85rem;
         padding: .15rem .6rem; border-radius: 10px; border: 1px dashed currentColor;
         background: rgba(255,255,255,.04); animation: twinkle 3.2s ease-in-out infinite; }
  .s1 { left: 1%;  top: 100px; color: #93c5fd; }
  .s2 { right: 2%; top: 56px;  color: #f0abfc; animation-delay: .7s; }
  .s3 { right: 17%; top: 4px;  color: #6ee7b7; animation-delay: 1.3s; }
  .s4 { left: 15%; top: 4px;   color: #fcd34d; animation-delay: 1.9s; }
  .s5 { right: 1%; bottom: 80px; color: #f9a8d4; animation-delay: .5s; }
  .s6 { left: 11%; bottom: 82px; color: #c4b5fd; animation-delay: 1.5s; }
</style>
</head>
<body>
<div class="stage">

  <!-- Neural network -->
  <svg class="net" viewBox="0 0 260 165" xmlns="http://www.w3.org/2000/svg">
    <g class="edges">
      <line x1="30" y1="30" x2="130" y2="20"/><line x1="30" y1="30" x2="130" y2="60"/>
      <line x1="30" y1="30" x2="130" y2="100"/><line x1="30" y1="30" x2="130" y2="140"/>
      <line x1="30" y1="75" x2="130" y2="20"/><line x1="30" y1="75" x2="130" y2="60"/>
      <line x1="30" y1="75" x2="130" y2="100"/><line x1="30" y1="75" x2="130" y2="140"/>
      <line x1="30" y1="120" x2="130" y2="20"/><line x1="30" y1="120" x2="130" y2="60"/>
      <line x1="30" y1="120" x2="130" y2="100"/><line x1="30" y1="120" x2="130" y2="140"/>
      <line x1="130" y1="20" x2="230" y2="55"/><line x1="130" y1="20" x2="230" y2="105"/>
      <line x1="130" y1="60" x2="230" y2="55"/><line x1="130" y1="60" x2="230" y2="105"/>
      <line x1="130" y1="100" x2="230" y2="55"/><line x1="130" y1="100" x2="230" y2="105"/>
      <line x1="130" y1="140" x2="230" y2="55"/><line x1="130" y1="140" x2="230" y2="105"/>
    </g>
    <circle cx="30" cy="30" r="7" style="animation-delay:0s"/>
    <circle cx="30" cy="75" r="7" style="animation-delay:.3s"/>
    <circle cx="30" cy="120" r="7" style="animation-delay:.6s"/>
    <circle cx="130" cy="20" r="7" style="animation-delay:.9s"/>
    <circle cx="130" cy="60" r="7" style="animation-delay:1.2s"/>
    <circle cx="130" cy="100" r="7" style="animation-delay:1.5s"/>
    <circle cx="130" cy="140" r="7" style="animation-delay:1.8s"/>
    <circle class="out" cx="230" cy="55" r="8" style="animation-delay:.4s"/>
    <circle class="out" cx="230" cy="105" r="8" style="animation-delay:1s"/>
    <text x="10" y="160">Input</text>
    <text x="105" y="160">Hidden</text>
    <text x="205" y="160">Output</text>
  </svg>

  <!-- Neuron buddy 1 (purple) -->
  <svg class="buddy b1" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs><radialGradient id="g1" cx="40%" cy="35%" r="70%">
      <stop offset="0%" stop-color="#c4b5fd"/><stop offset="100%" stop-color="#6C63FF"/>
    </radialGradient></defs>
    <g stroke="#a78bfa" stroke-width="3" stroke-linecap="round" fill="none">
      <path d="M50 50 L18 22"/><path d="M50 50 L10 52"/><path d="M50 50 L24 86"/>
      <path d="M50 50 L84 22"/><path d="M50 50 L92 62"/>
    </g>
    <circle cx="18" cy="22" r="4.5" fill="#f0abfc"/><circle cx="10" cy="52" r="4.5" fill="#f0abfc"/>
    <circle cx="24" cy="86" r="4.5" fill="#f0abfc"/><circle cx="84" cy="22" r="4.5" fill="#f0abfc"/>
    <circle cx="92" cy="62" r="4.5" fill="#f0abfc"/>
    <circle cx="50" cy="50" r="22" fill="url(#g1)" stroke="#e9d5ff" stroke-width="2"/>
    <circle cx="42" cy="46" r="5.5" fill="#fff"/><circle cx="58" cy="46" r="5.5" fill="#fff"/>
    <circle cx="43.5" cy="47" r="2.6" fill="#1e1b4b"/><circle cx="59.5" cy="47" r="2.6" fill="#1e1b4b"/>
    <path d="M42 58 Q50 66 58 58" stroke="#fff" stroke-width="2.6" fill="none" stroke-linecap="round"/>
  </svg>

  <!-- Neuron buddy 2 (green/blue) -->
  <svg class="buddy b2" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs><radialGradient id="g2" cx="40%" cy="35%" r="70%">
      <stop offset="0%" stop-color="#6ee7b7"/><stop offset="100%" stop-color="#3b82f6"/>
    </radialGradient></defs>
    <g stroke="#60a5fa" stroke-width="3" stroke-linecap="round" fill="none">
      <path d="M50 50 L16 26"/><path d="M50 50 L8 56"/><path d="M50 50 L28 88"/>
      <path d="M50 50 L86 20"/><path d="M50 50 L92 58"/>
    </g>
    <circle cx="16" cy="26" r="4.5" fill="#fcd34d"/><circle cx="8" cy="56" r="4.5" fill="#fcd34d"/>
    <circle cx="28" cy="88" r="4.5" fill="#fcd34d"/><circle cx="86" cy="20" r="4.5" fill="#fcd34d"/>
    <circle cx="92" cy="58" r="4.5" fill="#fcd34d"/>
    <circle cx="50" cy="50" r="22" fill="url(#g2)" stroke="#d1fae5" stroke-width="2"/>
    <circle cx="42" cy="46" r="5.5" fill="#fff"/><circle cx="58" cy="46" r="5.5" fill="#fff"/>
    <circle cx="43.5" cy="47" r="2.6" fill="#0c1a3d"/><circle cx="59.5" cy="47" r="2.6" fill="#0c1a3d"/>
    <path d="M42 58 Q50 66 58 58" stroke="#fff" stroke-width="2.6" fill="none" stroke-linecap="round"/>
  </svg>

  <!-- Floating emojis -->
  <span class="emo e1">🧠</span>
  <span class="emo e2">🕸️</span>
  <span class="emo e3">⚛️</span>
  <span class="emo e4">🤖</span>
  <span class="emo e5">🧬</span>
  <span class="emo e6">📊</span>
  <span class="emo e7">🔗</span>

  <!-- ML symbols -->
  <span class="sym s1">f(x)</span>
  <span class="sym s2">∑</span>
  <span class="sym s3">σ</span>
  <span class="sym s4">ReLU</span>
  <span class="sym s5">ŷ</span>
  <span class="sym s6">∇L</span>

</div>
</body>
</html>
"""


def render_ml_panel(height: int = 220):
    """Robot ke neeche neuron characters, neural network aur ML symbols ka panel."""
    components.html(ML_PANEL_HTML, height=height, scrolling=False)


apply_home_css()


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
        _html(
            f"""
<div class="hero-badge">🚀 Unified AI Intelligence Platform</div>
<div class="hero-head">{logo_html}<div class="hero-title left">TriMind AI</div></div>
<div class="hero-slogan left">{SLOGAN}</div>
<div class="hero-desc">
✨ Welcome to <b class="brand">TriMind AI</b> — a unified intelligence platform
that turns raw business data into decisions. Pick a room below:
🔎 <span class="hl-past">analyse the past</span>,
🔮 <span class="hl-future">predict the future</span>, and
📥 <span class="hl-report">download your own AI report</span>.
</div>
<div class="chip-row">
<span class="chip c-sales">📈 Sales Forecasting</span>
<span class="chip c-churn">👥 Churn Prediction</span>
<span class="chip c-hr">🧑‍💼 HR Attrition</span>
<span class="chip c-models">🤖 3 AI Models</span>
<span class="chip c-report">📄 Auto Reports</span>
</div>
<div class="hero-stats">
<div class="hero-stat"><div class="bar" style="background-image:linear-gradient(90deg,#34d399,#60a5fa);"></div>
<span class="ico">🏛️</span><b style="background-image:linear-gradient(90deg,#34d399,#60a5fa);">3</b><span class="lbl">Intelligence Rooms</span></div>
<div class="hero-stat"><div class="bar" style="background-image:linear-gradient(90deg,#a78bfa,#f0abfc);"></div>
<span class="ico">🧠</span><b style="background-image:linear-gradient(90deg,#a78bfa,#f0abfc);">3</b><span class="lbl">Models per Room</span></div>
<div class="hero-stat"><div class="bar" style="background-image:linear-gradient(90deg,#fbbf24,#f472b6);"></div>
<span class="ico">⚡</span><b style="background-image:linear-gradient(90deg,#fbbf24,#f472b6);">1-click</b><span class="lbl">AI Report</span></div>
</div>
"""
        )

    with h_right:
        with st.container(key="hero_robot"):
            show_robot(height=260)
        render_ml_panel(height=220)


# ---------------- Three Intelligence Rooms ----------------

section_head(
    "🧠 Choose Your Intelligence Room",
    "Har room mein 3 AI models, live predictions aur downloadable report.",
)

r1, r2, r3 = st.columns(3)


# Sales Analytics
with r1:
    _html(
        """
<div class="room-card sales">
<div class="bar"></div>
<div class="room-icon">📈</div>
<h3>Sales Analytics &amp; Profit Prediction</h3>
<p>Forecast revenue trends and profitability with regression models —
compare Linear Regression, Random Forest and Neural Networks.</p>
<div class="room-tags"><span>Regression</span><span>Forecast</span><span>Profit</span></div>
</div>
"""
    )

    if st.button(
        "Enter Sales Room →",
        key="go_sales",
        use_container_width=True,
    ):
        st.switch_page("pages/1_Sales_Analytics.py")


# Customer Churn
with r2:
    _html(
        """
<div class="room-card churn">
<div class="bar"></div>
<div class="room-icon">👥</div>
<h3>Customer Churn Prediction</h3>
<p>Identify at-risk customers before they leave — behaviour analysis,
retention insights and churn probability scores.</p>
<div class="room-tags"><span>Classification</span><span>Retention</span><span>Risk Score</span></div>
</div>
"""
    )

    if st.button(
        "Enter Churn Room →",
        key="go_churn",
        use_container_width=True,
    ):
        st.switch_page("pages/2_Customer_Churn.py")


# HR Analytics
with r3:
    _html(
        """
<div class="room-card hr">
<div class="bar"></div>
<div class="room-icon">🧑‍💼</div>
<h3>HR Analytics &amp; Employee Attrition</h3>
<p>Predict which employees are likely to leave — workforce retention
forecasts and satisfaction-driven insights.</p>
<div class="room-tags"><span>Attrition</span><span>Workforce</span><span>Satisfaction</span></div>
</div>
"""
    )

    if st.button(
        "Enter HR Room →",
        key="go_hr",
        use_container_width=True,
    ):
        st.switch_page("pages/3_HR_Analytics.py")


# ---------------- How It Works ----------------

section_head(
    "⚙️ How It Works",
    "Data se decision tak sirf 4 aasan steps.",
)

w1, w2, w3, w4 = st.columns(4)

steps = [
    (w1, "1", "📥", "Load Data", "Use the built-in sample dataset or upload your own CSV."),
    (w2, "2", "🧹", "Clean & Explore", "Automated cleaning plus interactive EDA charts."),
    (w3, "3", "🤖", "Train AI Models", "Three models compete — the best one wins, fairly."),
    (w4, "4", "🔮", "Predict & Report", "Live predictions plus a downloadable AI report."),
]

for col, num, emoji, title, description in steps:
    with col:
        _html(
            f"""
<div class="step-card">
<div class="step-num">{num}</div>
<div class="step-emoji">{emoji}</div>
<div class="step-title">{title}</div>
<div class="step-desc">{description}</div>
</div>
"""
        )


# ---------------- Footer ----------------

render_footer()