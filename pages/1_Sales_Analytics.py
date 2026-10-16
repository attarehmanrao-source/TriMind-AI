#pages/1_Sales_Analytics.py
"""Room 1 page — Sales Analytics & Profit Prediction."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import sales
from src.utils import (apply_custom_css, build_html_report, fig_to_base64,
                       play_voice, render_footer, render_sidebar, show_robot,
                       validate_upload)

st.set_page_config(page_title="Sales Analytics — TriMind AI", page_icon="📈",
                   layout="wide")
apply_custom_css()
render_sidebar()

PALETTE = ["#8b5cf6", "#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#fb923c"]


# ---------------- Helpers ----------------
def _html(markup: str):
    """Indent hata kar render karta hai, taake Markdown HTML ko code block na bana de."""
    clean = "\n".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(clean, unsafe_allow_html=True)


def apply_page_css():
    _html(
        """
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
        radial-gradient(520px 200px at 10% 0%, rgba(52,211,153,0.16), transparent 70%),
        radial-gradient(520px 200px at 95% 100%, rgba(139,92,246,0.22), transparent 70%),
        #161A27;
    animation: fadeUp 0.6s ease both;
}
.pg-icon {
    width: 64px; height: 64px; border-radius: 18px; font-size: 2rem;
    display: flex; align-items: center; justify-content: center;
    background: rgba(52,211,153,0.16); border: 1px solid rgba(52,211,153,0.4);
}
.pg-title {
    font-size: 2rem; font-weight: 800; line-height: 1.15; margin: 0;
    background: linear-gradient(90deg, #6ee7b7, #60a5fa, #c4b5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
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
div[data-testid="stPlotlyChart"], div[data-testid="stImage"] img {
    border-radius: 16px;
}
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

/* ---------- Predict result ---------- */
.pred-card {
    text-align: center; padding: 1.4rem 1rem; border-radius: 20px; margin-top: 1rem;
    border: 1px solid rgba(251,191,36,0.5);
    background: linear-gradient(145deg, rgba(251,191,36,0.14), rgba(244,114,182,0.1));
    animation: fadeUp 0.5s ease both;
}
.pred-card .pl { color: #FDE68A; font-size: 0.95rem; font-weight: 600; }
.pred-card .pv {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #fbbf24, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
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
    )


def page_header():
    _html(
        """
<div class="pg-header">
<div class="pg-icon">📈</div>
<div>
<div class="pg-title">Sales Analytics &amp; Profit Prediction</div>
<div class="pg-sub">Regression room — forecast profit from orders, discounts and segments.</div>
<div class="pg-badges"><span>Linear Regression</span><span>Random Forest</span><span>Neural Network</span></div>
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


def style_mpl(fig, ax):
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#3b3f63")
    ax.tick_params(colors="white")
    ax.grid(axis="y", color="#8b5cf6", alpha=0.12)


apply_page_css()

# ---------------- Header ----------------
hc1, hc2 = st.columns([4, 1], vertical_alignment="center")
with hc1:
    page_header()
with hc2:
    show_robot(height=120)

# ---------------- Data source ----------------
section(1, "Data Source", "Built-in sample data use karein ya apni CSV upload karein.")
src_choice = st.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True,
                      label_visibility="collapsed")
df = None
if src_choice == "Sample data":
    play_voice("loading")
    with st.status("Loading sample sales data…", expanded=True) as s:
        df = sales.load_data()
        s.update(label=f"Loaded {len(df):,} rows", state="complete")
else:
    up = st.file_uploader("Upload a sales CSV", type=["csv"])
    if up:
        raw = pd.read_csv(up, encoding="latin1")
        ok, msg = validate_upload(raw, sales.REQUIRED_COLS)
        if not ok:
            st.error(msg)
        else:
            df = raw
            st.success("Upload validated ✅")

if df is None:
    st.info("Choose a data source to begin.")
    render_footer()
    st.stop()

# ---------------- Clean ----------------
play_voice("cleaning")
with st.status("Cleaning data…", expanded=True) as s:
    df = sales.clean_data(df)
    s.update(label=f"Cleaned: {df.shape[0]:,} rows × {df.shape[1]} columns",
             state="complete")
st.toast("Data cleaned ✅", icon="🧹")

# ---------------- EDA ----------------
play_voice("analyzing")
section(2, "Exploratory Analysis", "Sales aur profit ka pehla jaiza.")
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "💵", f'${df["Sales"].sum():,.0f}', "Total Sales", "linear-gradient(90deg,#34d399,#60a5fa)")
kpi(k2, "💰", f'${df["Profit"].sum():,.0f}', "Total Profit", "linear-gradient(90deg,#a78bfa,#f0abfc)")
kpi(k3, "🧾", f'{df["Profit"].mean():,.1f}', "Avg Profit / Order", "linear-gradient(90deg,#fbbf24,#f472b6)")
kpi(k4, "✅", f'{(df["Profit"] > 0).mean() * 100:.1f}%', "Profitable Orders", "linear-gradient(90deg,#60a5fa,#a78bfa)")

st.markdown("")
e1, e2 = st.columns(2)
with e1:
    st.plotly_chart(style_fig(px.bar(df.groupby("Category", as_index=False)["Profit"].sum(),
                                     x="Category", y="Profit", color="Category",
                                     title="Profit by Category")), use_container_width=True)
with e2:
    st.plotly_chart(style_fig(px.bar(df.groupby("Region", as_index=False)["Profit"].sum(),
                                     x="Region", y="Profit", color="Region",
                                     title="Profit by Region")), use_container_width=True)

e3, e4 = st.columns(2)
with e3:
    monthly = df.groupby(["OrderYear", "OrderMonth"], as_index=False)["Profit"].sum()
    monthly["Period"] = monthly["OrderYear"].astype(str) + "-" + monthly["OrderMonth"].astype(str).str.zfill(2)
    st.plotly_chart(style_fig(px.line(monthly, x="Period", y="Profit",
                                      title="Profit Trend Over Time")), use_container_width=True)
with e4:
    st.plotly_chart(style_fig(px.scatter(df.sample(min(2000, len(df))), x="Sales", y="Profit",
                                         color="Category", opacity=0.6,
                                         title="Sales vs Profit")), use_container_width=True)

# Matplotlib chart (NumPy + Matplotlib usage)
fig, ax = plt.subplots(figsize=(6, 3.2))
cats = df["Category"].unique()
means = [df.loc[df["Category"] == c, "Discount"].mean() for c in cats]
ax.bar(cats, means, color=PALETTE[: len(cats)])
ax.set_title("Average Discount by Category", color="white")
style_mpl(fig, ax)
st.pyplot(fig)
mpl_b64 = fig_to_base64(fig)

# ---------------- Train ----------------
play_voice("training")
section(3, "Train Models", "3 regression models train hongay aur best model chuna jayega.")
if st.button("🚀 Train 3 Models", type="primary"):
    X_train, X_test, y_train, y_test = sales.split(df)
    prog = st.progress(0, text="Training Linear Regression…")
    with st.status("Training models…", expanded=True) as s:
        st.write("Step 1/3 — Linear Regression")
        pipes = {}
        from sklearn.pipeline import Pipeline
        for i, (name, model) in enumerate(sales.get_models().items()):
            st.write(f"Step {i + 1}/3 — {name}")
            pipe = Pipeline([("prep", sales._preprocessor()), ("model", model)])
            pipe.fit(X_train, y_train)
            pipes[name] = pipe
            prog.progress((i + 1) / 3, text=f"Trained {name}")
        s.update(label="All models trained ✅", state="complete")
    prog.empty()
    results = sales.evaluate(pipes, X_test, y_test)
    st.session_state["sales_pipes"] = pipes
    st.session_state["sales_results"] = results
    st.session_state["sales_splits"] = (X_train, X_test, y_train, y_test)
    st.toast("Training complete ✅", icon="🤖")
    play_voice("complete")
    st.rerun()

if "sales_results" not in st.session_state:
    st.info("Press **Train 3 Models** to compare regressors.")
    render_footer()
    st.stop()

results = st.session_state["sales_results"]
pipes = st.session_state["sales_pipes"]
X_train, X_test, y_train, y_test = st.session_state["sales_splits"]

section(4, "Model Comparison", "Teeno models ka muqabla — R² jitna zyada, RMSE jitna kam behtar.")
st.dataframe(results, use_container_width=True)
best_name = results.iloc[0]["Model"]
_html(
    f"""
<div class="best-banner"><div class="trophy">🏆</div>
<div><div class="bn">Best model: {best_name}</div>
<div class="bs">R² = {results.iloc[0]['R²']}</div></div></div>
"""
)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(style_fig(px.bar(results, x="Model", y="R²", color="Model",
                                     title="R² Comparison", text="R²")), use_container_width=True)
with c2:
    st.plotly_chart(style_fig(px.bar(results, x="Model", y="RMSE", color="Model",
                                     title="RMSE Comparison (lower is better)", text="RMSE")),
                    use_container_width=True)

# Actual vs predicted (best model)
best_pipe = pipes[best_name]
pred_all = best_pipe.predict(X_test)
fig2, ax2 = plt.subplots(figsize=(6, 3.4))
ax2.scatter(y_test, pred_all, alpha=0.35, color="#60a5fa", s=10)
lims = [min(y_test.min(), pred_all.min()), max(y_test.max(), pred_all.max())]
ax2.plot(lims, lims, "--", color="#f472b6", label="Perfect prediction")
ax2.set_xlabel("Actual Profit", color="white")
ax2.set_ylabel("Predicted Profit", color="white")
ax2.set_title(f"Actual vs Predicted — {best_name}", color="white")
ax2.legend(facecolor="#161A27", edgecolor="#3b3f63", labelcolor="white")
style_mpl(fig2, ax2)
st.pyplot(fig2)
avp_b64 = fig_to_base64(fig2)

# ---------------- Predict ----------------
section(5, "Predict Profit", "Order ki details dein aur best model se profit ka andaza lagayein.")
defaults = sales.feature_defaults(df)
options = sales.category_options(df)
with st.form("sales_predict"):
    cols = st.columns(3)
    inputs = {}
    for i, c in enumerate(sales.NUMERIC):
        with cols[i % 3]:
            inputs[c] = st.number_input(c, value=float(defaults[c]))
    for i, c in enumerate(sales.CATEGORICAL):
        with cols[i % 3]:
            opts = options[c]
            idx = opts.index(defaults[c]) if defaults[c] in opts else 0
            inputs[c] = st.selectbox(c, opts, index=idx)
    submitted = st.form_submit_button("Predict Profit 💰", type="primary")

if submitted:
    row = pd.DataFrame([{c: inputs[c] for c in sales.NUMERIC + sales.CATEGORICAL}])
    profit = float(best_pipe.predict(row)[0])
    _html(
        f"""
<div class="pred-card"><div class="pl">💰 Predicted Profit</div>
<div class="pv">${profit:,.2f}</div></div>
"""
    )
    st.toast(f"Predicted profit: ${profit:,.2f}", icon="💰")

# ---------------- Report ----------------
section(6, "Download Report", "Poori analysis ki HTML report ek click mein.")
metrics = {r["Model"]: f"R²={r['R²']}, RMSE={r['RMSE']}, MAE={r['MAE']}"
           for _, r in results.iterrows()}
metrics["Best model"] = best_name
metrics["Rows analysed"] = f"{len(df):,}"
html = build_html_report(
    "Sales Analytics & Profit Prediction",
    "Regression models trained on retail order data to forecast profit.",
    metrics, [mpl_b64, avp_b64],
    "Predictions are decision support, not financial advice. Retrain when new data arrives.",
)
st.download_button("📥 Download Sales Report", html,
                   file_name="sales_report.html", mime="text/html",
                   use_container_width=True)

render_footer()