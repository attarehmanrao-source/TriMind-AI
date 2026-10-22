# pages/4_ML_Pipeline_Builder.py
"""Room 4 page — ML Pipeline Builder (upload any CSV -> train any model)."""
import streamlit as st

from src.pipeline_ui import render_pipeline_workflow
from src.ui import apply_page_css, page_header, section
from src.utils import apply_custom_css, render_footer, show_robot

st.set_page_config(page_title="ML Pipeline Builder — TriMind AI", page_icon="🧪", layout="wide")
apply_custom_css()
apply_page_css(accent_rgb="139,92,246")

# --- BACK BUTTON ---
b1, _ = st.columns([1, 5])
with b1:
    if st.button("⬅️ Back to Home", use_container_width=True):
        st.switch_page("app.py")

# ---------------- Header ----------------
hc1, hc2 = st.columns([4, 1], vertical_alignment="center")
with hc1:
    page_header(
        "🧪", "ML Pipeline Builder",
        "Koi bhi CSV upload karein — columns chunein, encode aur scale karein, model train karein.",
        badges=["CSV Upload", "Encoding", "Scaling", "Any Model", "Evaluation"],
        title_grad="linear-gradient(90deg,#c4b5fd,#60a5fa,#f0abfc)",
    )
with hc2:
    show_robot(height=120)

# ---------------- 1. Upload ----------------
section(1, "Upload Data", "Apni CSV file upload karein — ibtidai rows aur columns ki maloomat neeche dikhegi.")
up = st.file_uploader("Upload a CSV file", type=["csv"])

if up is None:
    st.info("Shuru karne ke liye CSV file upload karein.")
else:
    render_pipeline_workflow(up, key_prefix="mlp", start_num=2)

render_footer()