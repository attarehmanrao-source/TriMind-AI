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

st.markdown('<div class="hero-title" style="font-size:2.2rem">📈 Sales Analytics & Profit Prediction</div>',
            unsafe_allow_html=True)
st.caption("Regression room — forecast profit from orders, discounts and segments.")
show_robot(height=120)

# ---------------- Data source ----------------
src_choice = st.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True)
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
st.subheader("🔍 Exploratory Analysis")
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi-card"><div class="kpi-value">${df["Sales"].sum():,.0f}</div><div class="kpi-label">Total Sales</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><div class="kpi-value">${df["Profit"].sum():,.0f}</div><div class="kpi-label">Total Profit</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["Profit"].mean():,.1f}</div><div class="kpi-label">Avg Profit / Order</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-value">{(df["Profit"] > 0).mean() * 100:.1f}%</div><div class="kpi-label">Profitable Orders</div></div>', unsafe_allow_html=True)

e1, e2 = st.columns(2)
with e1:
    st.plotly_chart(px.bar(df.groupby("Category", as_index=False)["Profit"].sum(),
                           x="Category", y="Profit", color="Category",
                           title="Profit by Category"), use_container_width=True)
with e2:
    st.plotly_chart(px.bar(df.groupby("Region", as_index=False)["Profit"].sum(),
                           x="Region", y="Profit", color="Region",
                           title="Profit by Region"), use_container_width=True)

e3, e4 = st.columns(2)
with e3:
    monthly = df.groupby(["OrderYear", "OrderMonth"], as_index=False)["Profit"].sum()
    monthly["Period"] = monthly["OrderYear"].astype(str) + "-" + monthly["OrderMonth"].astype(str).str.zfill(2)
    st.plotly_chart(px.line(monthly, x="Period", y="Profit",
                            title="Profit Trend Over Time"), use_container_width=True)
with e4:
    st.plotly_chart(px.scatter(df.sample(min(2000, len(df))), x="Sales", y="Profit",
                               color="Category", opacity=0.6,
                               title="Sales vs Profit"), use_container_width=True)

# Matplotlib chart (NumPy + Matplotlib usage)
fig, ax = plt.subplots(figsize=(6, 3.2))
cats = df["Category"].unique()
means = [df.loc[df["Category"] == c, "Discount"].mean() for c in cats]
ax.bar(cats, means, color="#6C63FF")
ax.set_title("Average Discount by Category", color="white")
ax.tick_params(colors="white")
fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#0E1117")
st.pyplot(fig)
mpl_b64 = fig_to_base64(fig)

# ---------------- Train ----------------
play_voice("training")
st.subheader("🤖 Train Models")
if st.button("Train 3 Models", type="primary"):
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

st.subheader("🏆 Model Comparison")
st.dataframe(results, use_container_width=True)
best_name = results.iloc[0]["Model"]
st.success(f"Best model: **{best_name}** (R² = {results.iloc[0]['R²']})")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(results, x="Model", y="R²", color="Model",
                           title="R² Comparison", text="R²"), use_container_width=True)
with c2:
    st.plotly_chart(px.bar(results, x="Model", y="RMSE", color="Model",
                           title="RMSE Comparison (lower is better)", text="RMSE"),
                    use_container_width=True)

# Actual vs predicted (best model)
best_pipe = pipes[best_name]
pred_all = best_pipe.predict(X_test)
fig2, ax2 = plt.subplots(figsize=(6, 3.4))
ax2.scatter(y_test, pred_all, alpha=0.35, color="#60a5fa", s=10)
lims = [min(y_test.min(), pred_all.min()), max(y_test.max(), pred_all.max())]
ax2.plot(lims, lims, "r--", label="Perfect prediction")
ax2.set_xlabel("Actual Profit", color="white"); ax2.set_ylabel("Predicted Profit", color="white")
ax2.set_title(f"Actual vs Predicted — {best_name}", color="white")
ax2.tick_params(colors="white"); ax2.legend()
fig2.patch.set_facecolor("#0E1117"); ax2.set_facecolor("#0E1117")
st.pyplot(fig2)
avp_b64 = fig_to_base64(fig2)

# ---------------- Predict ----------------
st.subheader("🔮 Predict Profit")
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
    st.metric("Predicted Profit", f"${profit:,.2f}")
    st.toast(f"Predicted profit: ${profit:,.2f}", icon="💰")

# ---------------- Report ----------------
st.subheader("📥 Download Report")
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
st.download_button("Download Sales Report", html,
                   file_name="sales_report.html", mime="text/html",
                   use_container_width=True)

render_footer()