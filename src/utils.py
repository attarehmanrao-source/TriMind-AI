"""Shared helpers for TriMind AI: theme CSS, footer, voice, reports."""
import base64
import io
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import math

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
        .stApp {
            font-family: 'Poppins', sans-serif;
            background: #0E1117;
        }
        header[data-testid="stHeader"] { background: transparent; }
        footer { visibility: hidden; }
        
        /* ---------- HIDE STREAMLIT DEFAULT SIDEBAR ---------- */
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        
        .block-container { 
            padding-top: 2rem; 
            max-width: 100% !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }
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

        /* ---------- Hero panel ---------- */
        @keyframes floatY {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-10px); }
        }
        .st-key-hero_panel {
            position: relative; overflow: hidden;
            padding: 1.8rem 2rem;
            background: #0E1117;
            border: 1px solid rgba(139, 92, 246, 0.35);
            border-radius: 22px;
            box-shadow: 0 0 40px rgba(108, 99, 255, 0.18),
                        0 0 90px rgba(139, 92, 246, 0.08);
        }
        .st-key-hero_panel::before {
            content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #6C63FF, #a78bfa, #60a5fa, #f0abfc, #6C63FF);
            background-size: 300% 100%;
            animation: gradientShift 6s ease infinite;
        }
        .st-key-hero_robot iframe { animation: floatY 5s ease-in-out infinite; }

        .hero-badge {
            display: inline-block; padding: 0.3rem 0.9rem; margin-bottom: 0.8rem;
            border-radius: 999px; font-size: 0.8rem; font-weight: 600; color: #c4b5fd;
            background: rgba(139, 92, 246, 0.12);
            border: 1px solid rgba(139, 92, 246, 0.4);
        }
        .hero-head { display: flex; align-items: center; gap: 0.9rem; }
        .hero-title.left { text-align: left; font-size: 3.2rem; margin-bottom: 0.1rem; }
        .hero-slogan.left { text-align: left; margin-bottom: 1rem; }
        .hero-desc { color: #d5d8f5; font-size: 1rem; line-height: 1.7; margin-bottom: 1rem; }
        .hero-desc b { color: #c4b5fd; }
        .chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.2rem; }
        .chip {
            padding: 0.35rem 0.8rem; border-radius: 999px; font-size: 0.82rem;
            color: #e0e2ff; background: rgba(108, 99, 255, 0.14);
            border: 1px solid rgba(139, 92, 246, 0.35);
            transition: all 0.2s ease;
        }
        .chip:hover { background: rgba(139, 92, 246, 0.32); transform: translateY(-2px); }
        .hero-stats { display: flex; flex-wrap: wrap; gap: 1.8rem; }
        .hero-stat b { display: block; font-size: 1.6rem; color: #ffffff; line-height: 1.2; }
        .hero-stat span { font-size: 0.78rem; color: #b8bce6; }

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
    return  # سائیڈبار ڈس ایبل ہے


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


# ==========================================
# CACHING FIX: آواز کو بار بار پروسیس ہونے سے بچانے کے لیے کیشنگ
# ==========================================
@st.cache_data(show_spinner=False)
def _get_audio_base64(path: str, mtime: float) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def play_voice(name: str, autoplay: bool = True):
    path = f"assets/audio/{name}.mp3"
    if not os.path.exists(path):
        return
    try:
        b64 = _get_audio_base64(path, os.path.getmtime(path))
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


# ---------------- Animated room illustrations ----------------
_FONT = "Segoe UI,Roboto,Helvetica,Arial,sans-serif"

_SVG_DEFS = """
<defs>
  <linearGradient id="gP" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6C63FF"/><stop offset="1" stop-color="#c084fc"/></linearGradient>
  <linearGradient id="gBlue" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#60a5fa"/></linearGradient>
  <linearGradient id="gGreen" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#10b981"/><stop offset="1" stop-color="#6ee7b7"/></linearGradient>
  <linearGradient id="gRed" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f43f5e"/><stop offset="1" stop-color="#fda4af"/></linearGradient>
  <linearGradient id="gAmber" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f59e0b"/><stop offset="1" stop-color="#fde68a"/></linearGradient>
  <linearGradient id="gGold" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fde68a"/><stop offset="1" stop-color="#f59e0b"/></linearGradient>
  <linearGradient id="gLine" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#f0abfc"/><stop offset="1" stop-color="#60a5fa"/></linearGradient>
  <linearGradient id="gArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#a78bfa" stop-opacity=".45"/><stop offset="1" stop-color="#a78bfa" stop-opacity="0"/></linearGradient>
  <linearGradient id="gGlass" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffffff" stop-opacity=".12"/><stop offset="1" stop-color="#ffffff" stop-opacity=".03"/></linearGradient>
  <linearGradient id="gEdge" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffffff" stop-opacity=".4"/><stop offset="1" stop-color="#a78bfa" stop-opacity=".15"/></linearGradient>
  <linearGradient id="gSweep" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#60a5fa" stop-opacity="0"/><stop offset="1" stop-color="#60a5fa" stop-opacity=".55"/></linearGradient>
  <filter id="shadow" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#000" flood-opacity=".5"/></filter>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <filter id="blur" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="18"/></filter>
</defs>
"""

_SVG_COMMON_CSS = """
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
@keyframes pulse{0%{transform:scale(1);opacity:.75}100%{transform:scale(2);opacity:0}}
@keyframes twinkle{0%,100%{opacity:.12}50%{opacity:1}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes drift{0%,100%{transform:translate(0,0)}50%{transform:translate(8px,-6px)}}
.fl{animation:bob 4.5s ease-in-out infinite}.fl2{animation-delay:-1.5s}.fl3{animation-delay:-3s}
.pulse{transform-box:fill-box;transform-origin:center;animation:pulse 2.4s ease-out infinite}
.tw{animation:twinkle 2.6s ease-in-out infinite}
.orb{animation:drift 9s ease-in-out infinite}
"""

_BG_ORBS = (
    '<circle class="orb" cx="125" cy="95" r="40" fill="#6C63FF" opacity=".22" filter="url(#blur)"/>'
    '<circle class="orb" cx="300" cy="225" r="38" fill="#60a5fa" opacity=".16" filter="url(#blur)" style="animation-delay:-4s"/>'
)

def _t(x, y, txt, size=10, weight=400, fill="#ffffff", anchor="start", extra=""):
    return (
        f'<text x="{x}" y="{y}" font-family="{_FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" {extra}>{txt}</text>'
    )

def _card(x, y, w, h, content, cls="fl", rx=16):
    return (
        f'<g class="{cls}"><g filter="url(#shadow)">'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="#161833" fill-opacity=".74"/></g>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="url(#gGlass)" stroke="url(#gEdge)"/>'
        f"{content}</g>"
    )

def _avatar(cx, cy, r, grad):
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#{grad})"/>'
        f'<circle cx="{cx}" cy="{cy - r * 0.2:.1f}" r="{r * 0.3:.1f}" fill="#fff" fill-opacity=".95"/>'
        f'<path d="M{cx - r * 0.5:.1f} {cy + r * 0.58:.1f} a{r * 0.5:.1f} {r * 0.46:.1f} 0 0 1 {r:.1f} 0 z" fill="#fff" fill-opacity=".95"/>'
    )

def _smooth(points):
    p = points
    d = f"M{p[0][0]:.1f} {p[0][1]:.1f}"
    for i in range(len(p) - 1):
        p0 = p[i - 1] if i > 0 else p[i]
        p1, p2 = p[i], p[i + 1]
        p3 = p[i + 2] if i + 2 < len(p) else p[i + 1]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}"
    return d

def _svg_wrap(css: str, body: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">'
        f"{_SVG_DEFS}<style>{_SVG_COMMON_CSS}{css}</style>{_BG_ORBS}{body}</svg>"
    )


# =====================  SALES  =====================
_S_PTS = [(38, 214), (74, 200), (110, 208), (146, 176), (182, 184), (218, 150), (256, 118)]
_S_LINE = _smooth(_S_PTS)
_S_AREA = _S_LINE + " L256 240 L38 240 Z"

_SALES_CSS = """
.s-chart{animation:s-fade 6s linear infinite}
@keyframes s-fade{0%,88%{opacity:1}96%,100%{opacity:0}}
.s-line{stroke-dasharray:100;animation:s-draw 6s linear infinite}
@keyframes s-draw{0%{stroke-dashoffset:100}62%,100%{stroke-dashoffset:0}}
.s-reveal{transform-box:fill-box;transform-origin:0 50%;animation:s-rev 6s linear infinite}
@keyframes s-rev{0%{transform:scaleX(0)}62%,100%{transform:scaleX(1)}}
.s-mk{animation:s-mk 6s linear infinite}
@keyframes s-mk{0%{opacity:0}3%,86%{opacity:1}94%,100%{opacity:0}}
.s-t1{animation:s-t1 6s linear infinite}.s-t2{animation:s-t2 6s linear infinite}.s-t3{animation:s-t3 6s linear infinite}
@keyframes s-t1{0%,19%{opacity:1}22%,100%{opacity:0}}
@keyframes s-t2{0%,19%{opacity:0}22%,40%{opacity:1}43%,100%{opacity:0}}
@keyframes s-t3{0%,40%{opacity:0}43%,100%{opacity:1}}
.s-ring{transform:rotate(-90deg);transform-box:fill-box;transform-origin:center;stroke-dasharray:150.8;animation:s-ring 6s ease-in-out infinite}
@keyframes s-ring{0%{stroke-dashoffset:150.8}35%,88%{stroke-dashoffset:27.1}96%,100%{stroke-dashoffset:150.8}}
.s-bar{transform-box:fill-box;transform-origin:50% 100%;animation:s-grow 6s ease-in-out infinite backwards}
.s-b1{animation-delay:.1s}.s-b2{animation-delay:.2s}.s-b3{animation-delay:.3s}.s-b4{animation-delay:.4s}.s-b5{animation-delay:.5s}
@keyframes s-grow{0%{transform:scaleY(0)}22%,88%{transform:scaleY(1)}98%,100%{transform:scaleY(0)}}
"""

_S_GRID = "".join(
    f'<line x1="38" y1="{y}" x2="258" y2="{y}" stroke="#fff" stroke-opacity=".07" stroke-dasharray="3 5"/>'
    for y in (130, 170, 210)
)
_S_XLABELS = "".join(
    _t(38 + i * 43.6, 256, m, 10, 400, "#7d82b5", "middle") for i, m in enumerate(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
)
_S_BARS = "".join(
    f'<rect class="s-bar s-b{i}" x="{296 + 14 * i}" y="{252 - h}" width="9" height="{h}" rx="3" '
    f'fill="url(#{"gGreen" if i == 5 else "gBlue"})"/>'
    for i, h in enumerate([14, 22, 17, 28, 24, 34])
)

_S_MAIN = (
    _t(36, 60, "REVENUE", 10, 600, "#9aa0d0", extra='letter-spacing="1.2"')
    + _t(36, 90, "$48.2K", 25, 700, "#ffffff")
    + '<rect x="132" y="72" width="60" height="20" rx="10" fill="#34d399" fill-opacity=".16" stroke="#34d399" stroke-opacity=".8"/>'
    '<path d="M141 85 L145 78 L149 85 Z" fill="#34d399"/>'
    + _t(154, 86, "24%", 11, 700, "#34d399")
    + _S_GRID
    + '<clipPath id="sReveal"><rect class="s-reveal" x="38" y="96" width="220" height="150"/></clipPath>'
    '<g class="s-chart">'
    f'<path d="{_S_AREA}" fill="url(#gArea)" clip-path="url(#sReveal)"/>'
    f'<path class="s-line" d="{_S_LINE}" pathLength="100" fill="none" stroke="url(#gLine)" stroke-width="3.5" '
    'stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>'
    '<g class="s-mk"><g>'
    '<animateMotion dur="6s" repeatCount="indefinite" calcMode="linear" keyPoints="0;1;1" keyTimes="0;.62;1" '
    f'path="{_S_LINE}"/>'
    '<circle class="pulse" r="8" fill="#60a5fa" opacity=".5"/><circle r="5" fill="#fff" stroke="#60a5fa" stroke-width="2.2"/>'
    '<rect x="-27" y="-37" width="54" height="22" rx="7" fill="#0E1117" stroke="#a78bfa" stroke-opacity=".7"/>'
    '<path d="M-4 -15 L0 -10 L4 -15 Z" fill="#0E1117" stroke="#a78bfa" stroke-opacity=".7"/>'
    '<rect x="-5" y="-16" width="10" height="3" fill="#0E1117"/>'
    + _t(0, -22, "$12K", 11, 700, "#fff", "middle", 'class="s-t1"')
    + _t(0, -22, "$29K", 11, 700, "#fff", "middle", 'class="s-t2"')
    + _t(0, -22, "$48K", 11, 700, "#fff", "middle", 'class="s-t3"')
    + "</g></g></g>"
    '<line x1="38" y1="240" x2="258" y2="240" stroke="#a78bfa" stroke-opacity=".3"/>'
    + _S_XLABELS
)

_S_PROFIT = (
    _t(298, 60, "PROFIT", 10, 600, "#9aa0d0", extra='letter-spacing="1.2"')
    + '<circle cx="334" cy="98" r="24" fill="none" stroke="#fff" stroke-opacity=".1" stroke-width="8"/>'
    '<circle class="s-ring" cx="334" cy="98" r="24" fill="none" stroke="url(#gGreen)" stroke-width="8" stroke-linecap="round"/>'
    + _t(334, 103, "82%", 14, 700, "#fff", "middle")
    + _t(334, 132, "margin", 10, 400, "#7d82b5", "middle")
)

_S_ORDERS = (
    _t(298, 176, "ORDERS", 10, 600, "#9aa0d0", extra='letter-spacing="1.2"')
    + _t(298, 200, "1,284", 17, 700, "#fff")
    + '<path d="M365 196 L369 190 L373 196 Z" fill="#34d399"/>'
    + _S_BARS
)

_SALES_BODY = (
    _card(20, 36, 252, 228, _S_MAIN, "fl")
    + _card(284, 40, 100, 104, _S_PROFIT, "fl fl2")
    + _card(284, 154, 100, 110, _S_ORDERS, "fl fl3")
    + '<g class="fl fl2"><circle cx="276" cy="150" r="13" fill="url(#gGold)"/>'
    + _t(276, 156, "$", 16, 700, "#92400e", "middle")
    + "</g>"
    '<g class="fl fl3"><circle cx="22" cy="38" r="11" fill="url(#gGold)"/>'
    + _t(22, 43, "$", 14, 700, "#92400e", "middle")
    + "</g>"
    '<circle class="tw" cx="392" cy="150" r="2" fill="#fff"/><circle class="tw" cx="12" cy="200" r="2" fill="#fff" style="animation-delay:-1.2s"/>'
)

# =====================  CHURN  =====================
_CX, _CY = 118, 150

def _polar(a_deg, r):
    a = math.radians(a_deg)
    return round(_CX + r * math.cos(a), 1), round(_CY + r * math.sin(a), 1)

_C_DOTS = [
    (30, 70, "ok"), (80, 42, "ok"), (120, 84, "risk"), (170, 58, "ok"),
    (215, 86, "risk"), (262, 38, "ok"), (305, 72, "warn"), (340, 52, "ok"),
]

def _churn_dot(a, r, kind):
    x, y = _polar(a, r)
    delay = round(a / 360 * 5, 2)
    grad = {"ok": "gP", "risk": "gRed", "warn": "gAmber"}[kind]
    ring = ""
    if kind != "ok":
        col = "#fb7185" if kind == "risk" else "#fbbf24"
        ring = f'<circle class="pulse" cx="{x}" cy="{y}" r="9" fill="none" stroke="{col}" stroke-width="2" style="animation-delay:{delay}s"/>'
    return f'{ring}<g class="c-blip" style="animation-delay:{delay}s">{_avatar(x, y, 9, grad)}</g>'

_a0x, _a0y = _polar(-55, 100)
_C_WEDGE = f"M{_CX} {_CY} L{_a0x} {_a0y} A100 100 0 0 1 {_CX + 100} {_CY} Z"

_CHURN_CSS = f"""
.c-sweep{{transform-origin:{_CX}px {_CY}px;animation:spin 5s linear infinite}}
.c-blip{{transform-box:fill-box;transform-origin:center;animation:c-blip 5s ease-out infinite backwards}}
@keyframes c-blip{{0%{{opacity:.6;transform:scale(1)}}3%{{opacity:1;transform:scale(1.3)}}22%,100%{{opacity:.85;transform:scale(1)}}}}
.c-bar{{transform-box:fill-box;transform-origin:0 50%;animation:c-fill 6s ease-in-out infinite backwards}}
.c-r1{{animation-delay:.15s}}.c-r2{{animation-delay:.35s}}.c-r3{{animation-delay:.55s}}
@keyframes c-fill{{0%{{transform:scaleX(0)}}25%,88%{{transform:scaleX(1)}}98%,100%{{transform:scaleX(0)}}}}
.c-cta{{transform-box:fill-box;transform-origin:center;animation:c-cta 2.4s ease-in-out infinite}}
@keyframes c-cta{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.04)}}}}
"""

_C_ROWS = ""
for i, (pct, grad, col) in enumerate([(86, "gRed", "#fb7185"), (72, "gRed", "#fb7185"), (54, "gAmber", "#fbbf24")]):
    y0 = 92 + i * 46
    _C_ROWS += (
        _avatar(260, y0 + 14, 11, grad)
        + f'<rect x="278" y="{y0 + 5}" width="62" height="7" rx="3.5" fill="#fff" fill-opacity=".22"/>'
        f'<rect x="278" y="{y0 + 17}" width="42" height="6" rx="3" fill="#fff" fill-opacity=".1"/>'
        + _t(374, y0 + 18, f"{pct}%", 12, 700, col, "end")
        + f'<rect x="246" y="{y0 + 32}" width="128" height="7" rx="3.5" fill="#fff" fill-opacity=".08"/>'
        f'<rect class="c-bar c-r{i + 1}" x="246" y="{y0 + 32}" width="{round(128 * pct / 100, 1)}" height="7" rx="3.5" fill="url(#{grad})"/>'
    )

_C_RADAR = (
    f'<circle cx="{_CX}" cy="{_CY}" r="104" fill="#161833" fill-opacity=".55" stroke="url(#gEdge)"/>'
    + "".join(
        f'<circle cx="{_CX}" cy="{_CY}" r="{r}" fill="none" stroke="#fff" stroke-opacity=".11" stroke-dasharray="{"0" if r == 100 else "3 5"}"/>'
        for r in (100, 68, 36)
    )
    + f'<line x1="{_CX - 100}" y1="{_CY}" x2="{_CX + 100}" y2="{_CY}" stroke="#fff" stroke-opacity=".08"/>'
    f'<line x1="{_CX}" y1="{_CY - 100}" x2="{_CX}" y2="{_CY + 100}" stroke="#fff" stroke-opacity=".08"/>'
    f'<g class="c-sweep"><path d="{_C_WEDGE}" fill="url(#gSweep)"/>'
    f'<line x1="{_CX}" y1="{_CY}" x2="{_CX + 100}" y2="{_CY}" stroke="#93c5fd" stroke-width="1.6" stroke-opacity=".9"/></g>'
    + "".join(_churn_dot(*d) for d in _C_DOTS)
    + f'<circle cx="{_CX}" cy="{_CY}" r="15" fill="url(#gP)" filter="url(#glow)"/>'
    f'<path d="M{_CX} {_CY - 9} L{_CX + 8} {_CY - 6} V{_CY + 2} Q{_CX + 8} {_CY + 8} {_CX} {_CY + 11} Q{_CX - 8} {_CY + 8} {_CX - 8} {_CY + 2} V{_CY - 6} Z" fill="#fff"/>'
    f'<path d="M{_CX - 3.6} {_CY + 0.5} L{_CX - 0.8} {_CY + 3.4} L{_CX + 4} {_CY - 2.6}" fill="none" stroke="#6C63FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
)

_C_LIST = (
    '<rect x="246" y="58" width="74" height="20" rx="10" fill="#fb7185" fill-opacity=".16" stroke="#fb7185" stroke-opacity=".8"/>'
    '<circle cx="258" cy="68" r="3" fill="#fb7185"/><circle class="pulse" cx="258" cy="68" r="3" fill="#fb7185"/>'
    + _t(266, 72, "AT RISK", 10, 700, "#fb7185", extra='letter-spacing=".8"')
    + _C_ROWS
    + '<g class="c-cta"><rect x="246" y="228" width="128" height="20" rx="10" fill="url(#gGreen)"/>'
    + _t(310, 241.5, "Send retention offer", 10, 700, "#06281d", "middle")
    + "</g>"
)

_CHURN_BODY = (
    _C_RADAR
    + _card(232, 44, 156, 212, _C_LIST, "fl fl2")
    + '<circle class="tw" cx="14" cy="60" r="2" fill="#fff"/><circle class="tw" cx="392" cy="270" r="2" fill="#fff" style="animation-delay:-1.4s"/>'
)

# =====================  HR  =====================
_HX, _HY = 122, 150
_H_SPARK = _smooth([(254, 204), (272, 196), (290, 200), (308, 186), (326, 190), (344, 176), (360, 180), (376, 170)])
_H_SPARK_AREA = _H_SPARK + " L376 212 L254 212 Z"

def _hr_node(x, y, grad, up_cls, risk=False):
    ring = (
        '<circle class="pulse" r="12" fill="none" stroke="#fb923c" stroke-width="2" style="animation-duration:1.8s"/>'
        if risk
        else ""
    )
    return (
        f'<g transform="translate({x},{y})"><g class="{up_cls}">{ring}'
        f'<circle r="13" fill="none" stroke="#fff" stroke-opacity=".25"/>{_avatar(0, 0, 11, grad)}</g></g>'
    )

def _ring_pts(r, angles):
    return [(round(r * math.cos(math.radians(a)), 1), round(r * math.sin(math.radians(a)), 1)) for a in angles]

_IN = _ring_pts(46, (90, 210, 330))
_OUT = _ring_pts(86, (18, 90, 162, 234, 306))

_HR_CSS = """
.h-o1,.h-o2,.h-u1,.h-u2{transform-origin:0 0}
.h-o1{animation:spin 28s linear infinite}
.h-u1{animation:spin 28s linear infinite reverse}
.h-o2{animation:spin 42s linear infinite reverse}
.h-u2{animation:spin 42s linear infinite}
.h-ring{transform:rotate(-90deg);transform-box:fill-box;transform-origin:center;stroke-dasharray:150.8;animation:h-ring 6s ease-in-out infinite}
@keyframes h-ring{0%{stroke-dashoffset:150.8}35%,88%{stroke-dashoffset:132.7}96%,100%{stroke-dashoffset:150.8}}
.h-spark{stroke-dasharray:100;animation:h-draw 6s linear infinite}
@keyframes h-draw{0%{stroke-dashoffset:100}55%,90%{stroke-dashoffset:0}98%,100%{stroke-dashoffset:-100}}
.h-flag{animation:c-cta 2.4s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
@keyframes c-cta{0%,100%{transform:scale(1)}}50%{transform:scale(1.04)}}}
"""

_HR_NETWORK = (
    f'<g transform="translate({_HX},{_HY})">'
    '<g class="h-o2"><circle r="86" fill="none" stroke="#a78bfa" stroke-opacity=".32" stroke-dasharray="4 7"/>'
    + _hr_node(*_OUT[0], "gBlue", "h-u2" )
    + _hr_node(*_OUT[1], "gP", "h-u2")
    + _hr_node(*_OUT[2], "gAmber", "h-u2", risk=True)
    + _hr_node(*_OUT[3], "gGreen", "h-u2")
    + _hr_node(*_OUT[4], "gP", "h-u2")
    + "</g>"
    '<g class="h-o1"><circle r="46" fill="none" stroke="#60a5fa" stroke-opacity=".38" stroke-dasharray="3 6"/>'
    + "".join(
        f'<line x1="0" y1="0" x2="{x}" y2="{y}" stroke="#a78bfa" stroke-opacity=".4"/>'
        f'<circle r="2.6" fill="#fff"><animateMotion dur="2.4s" begin="{i * 0.6}s" repeatCount="indefinite" path="M0 0 L{x} {y}"/></circle>'
        for i, (x, y) in enumerate(_IN)
    )
    + _hr_node(*_IN[0], "gP", "h-u1")
    + _hr_node(*_IN[1], "gGreen", "h-u1")
    + _hr_node(*_IN[2], "gBlue", "h-u1")
    + "</g>"
    '<circle class="pulse" r="22" fill="none" stroke="#a78bfa" stroke-width="2"/>'
    '<circle r="22" fill="url(#gP)" filter="url(#glow)"/>'
    '<circle cx="-9" cy="-1" r="3.4" fill="#fff" fill-opacity=".7"/><path d="M-15 11 Q-15 4 -9 4 Q-4 4 -3 9 Z" fill="#fff" fill-opacity=".7"/>'
    '<circle cx="9" cy="-1" r="3.4" fill="#fff" fill-opacity=".7"/><path d="M15 11 Q15 4 9 4 Q4 4 3 9 Z" fill="#fff" fill-opacity=".7"/>'
    '<circle cy="-5" r="4.6" fill="#fff"/><path d="M-9 12 Q-9 2 0 2 Q9 2 9 12 Z" fill="#fff"/>'
    "</g>"
)

_HR_CARD = (
    _t(256, 66, "ATTRITION", 10, 600, "#9aa0d0", extra='letter-spacing="1.2"')
    + '<circle cx="282" cy="108" r="24" fill="none" stroke="#fff" stroke-opacity=".1" stroke-width="8"/>'
    '<circle cx="282" cy="108" r="24" fill="none" stroke="#34d399" stroke-width="8" stroke-opacity=".35"/>'
    '<circle class="h-ring" cx="282" cy="108" r="24" fill="none" stroke="url(#gRed)" stroke-width="8" stroke-linecap="round"/>'
    + _t(282, 113, "12%", 13, 700, "#fff", "middle")
    + '<circle cx="320" cy="94" r="3.2" fill="#34d399"/>' + _t(328, 98, "Stay 88%", 10, 500, "#cfd3f5")
    + '<circle cx="320" cy="110" r="3.2" fill="#fbbf24"/>' + _t(328, 114, "Risk 9%", 10, 500, "#cfd3f5")
    + '<circle cx="320" cy="126" r="3.2" fill="#fb7185"/>' + _t(328, 130, "Left 3%", 10, 500, "#cfd3f5")
    + _t(256, 156, "SATISFACTION", 10, 600, "#9aa0d0", extra='letter-spacing="1.2"')
    + _t(376, 156, "4.2/5", 12, 700, "#34d399", "end")
    + f'<path d="{_H_SPARK_AREA}" fill="url(#gArea)" opacity=".8"/>'
    f'<path class="h-spark" d="{_H_SPARK}" pathLength="100" fill="none" stroke="url(#gLine)" stroke-width="3" '
    'stroke-linecap="round" filter="url(#glow)"/>'
    '<g class="h-flag"><rect x="252" y="224" width="128" height="22" rx="11" fill="#fb923c" fill-opacity=".15" stroke="#fb923c" stroke-opacity=".85"/>'
    '<circle cx="265" cy="235" r="3.4" fill="#fb923c"/><circle class="pulse" cx="265" cy="235" r="3.4" fill="#fb923c"/>'
    + _t(274, 238.3, "3 employees flagged", 9.5, 600, "#fdba74")
    + "</g>"
)

_HR_BODY = (
    _HR_NETWORK
    + _card(240, 44, 150, 212, _HR_CARD, "fl fl2")
    + '<circle class="tw" cx="16" cy="48" r="2" fill="#fff"/><circle class="tw" cx="226" cy="274" r="2.2" fill="#fff" style="animation-delay:-1s"/>'
    '<circle class="tw" cx="18" cy="262" r="2" fill="#fff" style="animation-delay:-.5s"/>'
)

ROOM_SCENES = {
    "sales": _svg_wrap(_SALES_CSS, _SALES_BODY),
    "churn": _svg_wrap(_CHURN_CSS, _CHURN_BODY),
    "hr": _svg_wrap(_HR_CSS, _HR_BODY),
}


# =====================  AI DATA-SCIENCE ART  =====================
_M_SPL = ".4 0 .2 1;0 0 1 1;.4 0 .2 1"

def _anim(attr, a, b, kt="0;.5;.9;1"):
    return (
        f'<animate attributeName="{attr}" values="{a};{b};{b};{a}" keyTimes="{kt}" dur="8s" '
        f'repeatCount="indefinite" calcMode="spline" keySplines="{_M_SPL}"/>'
    )

def _pill(x, y, w, txt, col, size=10):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="20" rx="10" fill="{col}" fill-opacity=".16" stroke="{col}" stroke-opacity=".8"/>'
        + _t(x + w / 2, y + 14, txt, size, 700, col, "middle")
    )

def _head(txt):
    return _t(34, 52, txt, 10, 600, "#9aa0d0", extra='letter-spacing="1.2"')


_M_CSS = """
@keyframes m-s1{0%,24%{opacity:1}28%,100%{opacity:0}}
@keyframes m-s2{0%,24%{opacity:0}28%,50%{opacity:1}54%,100%{opacity:0}}
@keyframes m-s3{0%,50%{opacity:0}54%,92%{opacity:1}96%,100%{opacity:0}}
.m-s1{animation:m-s1 8s linear infinite}.m-s2{animation:m-s2 8s linear infinite}.m-s3{animation:m-s3 8s linear infinite}
@keyframes m-halo{0%,40%{opacity:0}58%,90%{opacity:1}100%{opacity:0}}
.m-halo{animation:m-halo 8s ease-in-out infinite}
@keyframes m-loss{0%{transform:scaleX(1)}55%,92%{transform:scaleX(.16)}100%{transform:scaleX(1)}}
.m-loss{transform-box:fill-box;transform-origin:0 50%;animation:m-loss 8s ease-in-out infinite}
.t-d1{stroke-dasharray:100;animation:t-d1 8s ease-in-out infinite}
@keyframes t-d1{0%,6%{stroke-dashoffset:100}18%,90%{stroke-dashoffset:0}98%,100%{stroke-dashoffset:100}}
.t-d2{stroke-dasharray:100;animation:t-d2 8s ease-in-out infinite}
@keyframes t-d2{0%,24%{stroke-dashoffset:100}36%,90%{stroke-dashoffset:0}98%,100%{stroke-dashoffset:100}}
.t-p1{transform-box:fill-box;transform-origin:center;animation:t-p1 8s ease-out infinite}
@keyframes t-p1{0%,12%{opacity:0;transform:scale(.3)}18%{opacity:1;transform:scale(1.2)}22%,90%{opacity:1;transform:scale(1)}98%,100%{opacity:0;transform:scale(.3)}}
.t-p2{transform-box:fill-box;transform-origin:center;animation:t-p2 8s ease-out infinite}
@keyframes t-p2{0%,30%{opacity:0;transform:scale(.3)}36%{opacity:1;transform:scale(1.2)}40%,90%{opacity:1;transform:scale(1)}98%,100%{opacity:0;transform:scale(.3)}}
.t-dot{animation:t-dot 8s linear infinite}
@keyframes t-dot{0%,52%{opacity:0}56%,84%{opacity:1}88%,100%{opacity:0}}
"""

_RKT = "0;.55;.92;1"
_R_PTS = [(50 + i * 22, round(215 - 0.42 * (i * 22) + 18 * math.sin(i * 1.9), 1)) for i in range(14)]

def _ly(px, y1, y2):
    return round(y1 + (y2 - y1) * (px - 40) / 320, 1)

_R_BODY_IN = (
    _head("AI MODEL · LEARNING")
    + "".join(
        f'<line x1="40" y1="{y}" x2="360" y2="{y}" stroke="#fff" stroke-opacity=".06" stroke-dasharray="3 5"/>'
        for y in (100, 140, 180, 220)
    )
    + '<line x1="40" y1="244" x2="360" y2="244" stroke="#a78bfa" stroke-opacity=".3"/>'
    + "".join(
        f'<line x1="{px}" y1="{py}" x2="{px}" y2="{_ly(px, 219.2, 84.8)}" stroke="#fb7185" stroke-opacity=".65" stroke-width="1.6">'
        + _anim("y2", _ly(px, 120, 190), _ly(px, 219.2, 84.8), _RKT)
        + "</line>"
        for px, py in _R_PTS
    )
    + '<line x1="40" x2="360" stroke="#a78bfa" stroke-opacity=".28" stroke-width="9" stroke-linecap="round" y1="219.2" y2="84.8">'
    + _anim("y1", 120, 219.2, _RKT) + _anim("y2", 190, 84.8, _RKT) + "</line>"
    '<line x1="40" x2="360" stroke="#e9d5ff" stroke-width="2.6" stroke-linecap="round" y1="219.2" y2="84.8">'
    + _anim("y1", 120, 219.2, _RKT) + _anim("y2", 190, 84.8, _RKT) + "</line>"
    + "".join(
        f'<circle cx="{px}" cy="{py}" r="4.6" fill="#60a5fa" stroke="#fff" stroke-opacity=".85" stroke-width="1.4"/>'
        for px, py in _R_PTS
    )
    + _pill(294, 36, 76, "", "#34d399")
    + _t(332, 50, "R² 0.38", 10, 700, "#34d399", "middle", 'class="m-s1"')
    + _t(332, 50, "R² 0.81", 10, 700, "#34d399", "middle", 'class="m-s2"')
    + _t(332, 50, "R² 0.94", 10, 700, "#34d399", "middle", 'class="m-s3"')
    + _t(34, 264, "minimizing error", 9, 500, "#7d82b5")
    + _t(212, 264, "LOSS", 8, 700, "#9aa0d0", extra='letter-spacing="1"')
    + '<rect x="246" y="258" width="124" height="6" rx="3" fill="#fff" fill-opacity=".1"/>'
    '<rect class="m-loss" x="246" y="258" width="124" height="6" rx="3" fill="url(#gRed)"/>'
)

_AI_SALES = (
    _card(16, 24, 368, 252, _R_BODY_IN, "fl")
    + '<g class="fl fl2"><circle cx="388" cy="34" r="11" fill="url(#gGold)"/>' + _t(388, 39, "$", 14, 700, "#92400e", "middle") + "</g>"
    '<circle class="tw" cx="8" cy="150" r="2" fill="#fff"/><circle class="tw" cx="394" cy="270" r="2" fill="#fff" style="animation-delay:-1.2s"/>'
)

def _ai_cl(cx, cy, n, spread, seed):
    out = []
    for i in range(n):
        a = seed + i * 2.399
        r = 0.4 + 0.6 * (((i * 7 + seed) % 10) / 10)
        out.append((round(cx + spread * r * math.cos(a), 1), round(cy + spread * 0.8 * r * math.sin(a), 1)))
    return out

_CL = [("#34d399", (110, 150), (70, 210)), ("#fbbf24", (280, 125), (330, 200)), ("#fb7185", (195, 205), (210, 95))]

_cl_halos, _cl_pts, _cl_cent, _g = "", "", "", 0
for _k, (_col, (_fx, _fy), (_ix, _iy)) in enumerate(_CL):
    _cl_halos += (
        f'<circle class="m-halo" cx="{_fx}" cy="{_fy}" r="44" fill="{_col}" fill-opacity=".13" '
        f'stroke="{_col}" stroke-opacity=".55" stroke-dasharray="3 4"/>'
    )
    for (_x1, _y1) in _ai_cl(_fx, _fy, 6, 26, 3 + _k * 2):
        _x0, _y0 = 56 + (_g * 83) % 290, 84 + (_g * 47) % 150
        _cl_pts += (
            f'<circle r="5" cx="{_x0}" cy="{_y0}" fill="#9aa0d0" stroke="#fff" stroke-opacity=".6">'
            + _anim("cx", _x0, _x1) + _anim("cy", _y0, _y1) + _anim("fill", "#9aa0d0", _col) + "</circle>"
        )
        _g += 1
    _cl_cent += (
        f'<circle r="8.5" cx="{_ix}" cy="{_iy}" fill="{_col}" stroke="#fff" stroke-width="2.2">'
        + _anim("cx", _ix, _fx) + _anim("cy", _iy, _fy) + "</circle>"
    )

_K_IN = (
    _head("CLUSTERING · SEGMENTS")
    + "".join(
        f'<line x1="40" y1="{y}" x2="360" y2="{y}" stroke="#fff" stroke-opacity=".06" stroke-dasharray="3 5"/>'
        for y in (100, 140, 180, 220)
    )
    + _cl_halos + _cl_pts + _cl_cent
    + _pill(312, 36, 58, "k = 3", "#a78bfa")
    + _t(34, 264, "iteration 1", 9, 600, "#7d82b5", extra='class="m-s1"')
    + _t(34, 264, "iteration 3", 9, 600, "#fbbf24", extra='class="m-s2"')
    + _t(34, 264, "converged ✓", 9, 700, "#34d399", extra='class="m-s3"')
    + '<circle cx="196" cy="261" r="4" fill="#34d399"/>' + _t(204, 264, "Loyal", 9, 500, "#cfd3f5")
    + '<circle cx="246" cy="261" r="4" fill="#fbbf24"/>' + _t(254, 264, "Passive", 9, 500, "#cfd3f5")
    + '<circle cx="303" cy="261" r="4" fill="#fb7185"/>' + _t(311, 264, "At-risk", 9, 500, "#cfd3f5")
)

_AI_CHURN = (
    _card(16, 24, 368, 252, _K_IN, "fl")
    + '<circle class="tw" cx="8" cy="40" r="2" fill="#fff"/><circle class="tw" cx="394" cy="262" r="2" fill="#fff" style="animation-delay:-1.4s"/>'
)

_T_ROOT = (200, 100)
_T_L1 = [(125, 156), (275, 156)]
_T_L2 = [(88, 212), (162, 212), (238, 212), (312, 212)]
_T_LEAF = [("Stay", "#34d399"), ("Leave", "#fb7185"), ("Stay", "#34d399"), ("Leave", "#fb7185")]

def _seg(a, b, cls, col="#a78bfa"):
    return (
        f'<path class="{cls}" d="M{a[0]} {a[1]} L{b[0]} {b[1]}" pathLength="100" fill="none" '
        f'stroke="{col}" stroke-opacity=".7" stroke-width="2.2" stroke-linecap="round"/>'
    )

_T_LINES = (
    "".join(_seg(_T_ROOT, p, "t-d1") for p in _T_L1)
    + "".join(_seg(_T_L1[i // 2], p, "t-d2") for i, p in enumerate(_T_L2))
)
_T_NODES1 = '<g class="t-p1">' + "".join(
    f'<circle cx="{x}" cy="{y}" r="11" fill="url(#gBlue)" stroke="#fff" stroke-opacity=".5"/>' for x, y in _T_L1
) + "</g>"
_T_NODES2 = '<g class="t-p2">' + "".join(
    f'<rect x="{x - 23}" y="{y - 10}" width="46" height="20" rx="10" fill="{c}" fill-opacity=".18" stroke="{c}" stroke-opacity=".85"/>'
    + _t(x, y + 3.5, lab, 9, 700, c, "middle")
    for (x, y), (lab, c) in zip(_T_L2, _T_LEAF)
) + "</g>"
_T_PATH = "M200 100 L275 156 L312 212"

_H_IN = (
    _head("RANDOM FOREST · DECISION PATH")
    + _pill(298, 36, 72, "100 trees", "#a78bfa", 9)
    + _T_LINES
    + '<circle class="pulse" cx="200" cy="100" r="14" fill="none" stroke="#a78bfa" stroke-width="2"/>'
    + '<circle cx="200" cy="100" r="14" fill="url(#gP)" filter="url(#glow)"/>'
    + _t(200, 80, "Overtime?", 9, 600, "#cfd3f5", "middle")
    + _T_NODES1 + _T_NODES2
    + f'<path class="t-dot" d="{_T_PATH}" fill="none" stroke="#fb7185" stroke-width="3" stroke-linecap="round" stroke-opacity=".7"/>'
    + '<g class="t-dot"><circle r="9" fill="#fb7185" opacity=".35"/><circle r="4.5" fill="#fff">'
    f'<animateMotion dur="8s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;.55;.84;1" calcMode="linear" path="{_T_PATH}"/>'
    "</circle></g>"
    + '<circle cx="40" cy="262" r="4" fill="#34d399"/>' + _t(48, 265, "Stay", 9, 500, "#cfd3f5")
    + '<circle cx="90" cy="262" r="4" fill="#fb7185"/>' + _t(98, 265, "Leave", 9, 500, "#cfd3f5")
    + _t(366, 265, "prediction: Leave", 9, 700, "#fda4af", "end", 'class="t-dot"')
)

_AI_HR = (
    _card(16, 24, 368, 252, _H_IN, "fl")
    + '<circle class="tw" cx="8" cy="260" r="2" fill="#fff"/><circle class="tw" cx="394" cy="60" r="2" fill="#fff" style="animation-delay:-1s"/>'
)

AI_SCENES = {
    "sales": _svg_wrap(_M_CSS, _AI_SALES),
    "churn": _svg_wrap(_M_CSS, _AI_CHURN),
    "hr": _svg_wrap(_M_CSS, _AI_HR),
}


def _detect_room():
    f = sys._getframe(2)
    while f is not None:
        name = os.path.basename(f.f_code.co_filename).lower()
        if "sales" in name: return "sales"
        if "churn" in name: return "churn"
        if "hr_analytics" in name or name == "hr.py": return "hr"
        f = f.f_back
    return None


# ==========================================
# CACHING FIX: اینیمیشنز کو کیش کرنے کے لیے
# ==========================================
@st.cache_data(show_spinner=False)
def _room_scene_html(room: str, height: int) -> str:
    b64 = base64.b64encode(ROOM_SCENES[room].encode("utf-8")).decode()
    max_w = max(int(height * 1.5), 300)
    return (
        '<div style="display:flex;justify-content:center;">'
        f'<img src="data:image/svg+xml;base64,{b64}" alt="{room} animation" '
        f'style="width:100%;max-width:{max_w}px;"/></div>'
    )

@st.cache_data(show_spinner=False)
def _ai_art_html(room: str, max_width: int) -> str:
    b64 = base64.b64encode(AI_SCENES[room].encode("utf-8")).decode()
    return (
        '<div style="display:flex;justify-content:center;margin-top:0.4rem;">'
        f'<img src="data:image/svg+xml;base64,{b64}" alt="AI data science animation" '
        f'style="width:100%;max-width:{max_width}px;"/></div>'
    )

def show_ai_art(room=None, max_width: int = 330):
    room = room or _detect_room()
    if room in AI_SCENES:
        st.markdown(_ai_art_html(room, max_width), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _robot_transparent_b64(path: str, mtime: float, tol: int = 18) -> str:
    from PIL import Image, ImageDraw
    img = Image.open(path).convert("RGBA")
    img.thumbnail((400, 400))
    w, h = img.size
    for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        ImageDraw.floodfill(img, corner, (0, 0, 0, 0), thresh=tol)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def show_robot(height: int = 220, room=None):
    room = room or _detect_room()
    if room in ROOM_SCENES:
        st.markdown(_room_scene_html(room, height), unsafe_allow_html=True)
        show_ai_art(room)
        return

    try:
        from streamlit_lottie import st_lottie
        import requests

        @st.cache_data(show_spinner=False)
        def _load(url):
            r = requests.get(url, timeout=5)
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
    rows = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in metrics.items())
    imgs = "".join(f'<img src="data:image/png;base64,{b}" style="max-width:100%;margin:12px 0;border-radius:8px;"/>' for b in figures_b64)
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