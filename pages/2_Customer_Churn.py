#pages/2_Customer_Churn.py
"""Room 2 page — Customer Churn Prediction."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src import churn
from src.ui import (PALETTE, apply_page_css, best_banner, kpi, page_header,
                    result_card, section, style_fig, style_mpl)
from src.utils import (apply_custom_css, build_html_report, fig_to_base64,
                       play_voice, render_footer, render_sidebar, show_robot,
                       validate_upload)

st.set_page_config(page_title="Customer Churn — TriMind AI", page_icon="👥",
                   layout="wide")
apply_custom_css()
render_sidebar()
apply_page_css(accent_rgb="96,165,250")

STATUS_COLORS = {"Churned": "#f472b6", "Stayed": "#60a5fa"}

# ---------------- Header ----------------
hc1, hc2 = st.columns([4, 1], vertical_alignment="center")
with hc1:
    page_header(
        "👥", "Customer Churn Prediction",
        "Classification room — spot customers likely to leave, before they do.",
        badges=["Logistic Regression", "Random Forest", "Neural Network"],
        title_grad="linear-gradient(90deg,#93c5fd,#c4b5fd,#f0abfc)",
    )
with hc2:
    show_robot(height=120)

# ---------------- Data source ----------------
section(1, "Data Source", "Built-in sample data use karein ya apni CSV upload karein.")
src_choice = st.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True,
                      label_visibility="collapsed")
df = None
if src_choice == "Sample data":
    play_voice("loading")
    with st.status("Loading sample churn data…", expanded=True) as s:
        df = churn.load_data()
        s.update(label=f"Loaded {len(df):,} rows", state="complete")
else:
    up = st.file_uploader("Upload a churn CSV", type=["csv"])
    if up:
        raw = pd.read_csv(up)
        ok, msg = validate_upload(raw, churn.REQUIRED_COLS)
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
    df = churn.clean_data(df)
    s.update(label=f"Cleaned: {df.shape[0]:,} rows × {df.shape[1]} columns",
             state="complete")
st.toast("Data cleaned ✅", icon="🧹")

# ---------------- EDA ----------------
play_voice("analyzing")
section(2, "Exploratory Analysis", "Customers ke behaviour aur churn ka pehla jaiza.")
rate = df[churn.TARGET].mean() * 100
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "👥", f"{len(df):,}", "Customers", "linear-gradient(90deg,#60a5fa,#a78bfa)")
kpi(k2, "⚠️", f"{rate:.1f}%", "Churn Rate", "linear-gradient(90deg,#fb7185,#f472b6)")
kpi(k3, "⏳", f'{df["tenure"].mean():.1f}', "Avg Tenure (months)", "linear-gradient(90deg,#34d399,#60a5fa)")
kpi(k4, "💳", f'${df["MonthlyCharges"].mean():.2f}', "Avg Monthly Charges", "linear-gradient(90deg,#fbbf24,#f472b6)")

st.markdown("")
status_series = df[churn.TARGET].map({1: "Churned", 0: "Stayed"})
e1, e2 = st.columns(2)
with e1:
    counts = status_series.value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    st.plotly_chart(style_fig(px.pie(counts, names="Status", values="Count",
                                     title="Churn Distribution", hole=0.45,
                                     color="Status", color_discrete_map=STATUS_COLORS)),
                    use_container_width=True)
with e2:
    st.plotly_chart(style_fig(px.histogram(df, x="tenure", color=status_series,
                                           barmode="overlay", opacity=0.7,
                                           title="Tenure vs Churn",
                                           color_discrete_map=STATUS_COLORS)),
                    use_container_width=True)

e3, e4 = st.columns(2)
with e3:
    st.plotly_chart(style_fig(px.box(df, x=status_series, y="MonthlyCharges",
                                     color=status_series,
                                     title="Monthly Charges by Churn Status",
                                     color_discrete_map=STATUS_COLORS)),
                    use_container_width=True)
with e4:
    if "Contract" in df.columns:
        ct = df.groupby(["Contract", churn.TARGET]).size().reset_index(name="n")
        ct["Status"] = ct[churn.TARGET].map({1: "Churned", 0: "Stayed"})
        st.plotly_chart(style_fig(px.bar(ct, x="Contract", y="n", color="Status",
                                         barmode="group", title="Churn by Contract Type",
                                         color_discrete_map=STATUS_COLORS)),
                        use_container_width=True)

# Matplotlib: churn rate by tenure bucket
fig, ax = plt.subplots(figsize=(6, 3.2))
df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 100],
                             labels=["0-12", "13-24", "25-48", "49+"])
bucket_rate = df.groupby("tenure_bucket", observed=True)[churn.TARGET].mean() * 100
ax.bar(bucket_rate.index.astype(str), bucket_rate.values,
       color=PALETTE[: len(bucket_rate)])
ax.set_title("Churn Rate % by Tenure Bucket", color="white")
style_mpl(fig, ax)
st.pyplot(fig)
mpl_b64 = fig_to_base64(fig)

# ---------------- Train ----------------
play_voice("training")
section(3, "Train Models", "3 classification models train hongay aur best model chuna jayega.")
if st.button("🚀 Train 3 Models", type="primary"):
    X_train, X_test, y_train, y_test, num, cat = churn.split(df)
    prog = st.progress(0, text="Training…")
    with st.status("Training models…", expanded=True) as s:
        pipes = {}
        for i, (name, model) in enumerate(churn.get_models().items()):
            st.write(f"Step {i + 1}/3 — {name}")
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            prep = ColumnTransformer([
                ("num", StandardScaler(), num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
            pipe = Pipeline([("prep", prep), ("model", model)])
            pipe.fit(X_train, y_train)
            pipes[name] = pipe
            prog.progress((i + 1) / 3, text=f"Trained {name}")
        s.update(label="All models trained ✅", state="complete")
    prog.empty()
    results, probas = churn.evaluate(pipes, X_test, y_test)
    st.session_state["churn_pipes"] = pipes
    st.session_state["churn_results"] = results
    st.session_state["churn_splits"] = (X_test, y_test)
    st.toast("Training complete ✅", icon="🤖")
    play_voice("complete")
    st.rerun()

if "churn_results" not in st.session_state:
    st.info("Press **Train 3 Models** to compare classifiers.")
    render_footer()
    st.stop()

results = st.session_state["churn_results"]
pipes = st.session_state["churn_pipes"]
X_test, y_test = st.session_state["churn_splits"]

section(4, "Model Comparison", "Teeno models ka muqabla — F1 jitna zyada, behtar.")
st.dataframe(results, use_container_width=True)
best_name = results.iloc[0]["Model"]
best_banner(best_name, f"F1 = {results.iloc[0]['F1']}")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(style_fig(px.bar(results, x="Model", y="F1", color="Model",
                                     title="F1-Score Comparison", text="F1")),
                    use_container_width=True)
with c2:
    melt = results.melt(id_vars="Model", value_vars=["Precision", "Recall"],
                        var_name="Metric", value_name="Score")
    st.plotly_chart(style_fig(px.bar(melt, x="Model", y="Score", color="Metric",
                                     barmode="group", title="Precision vs Recall")),
                    use_container_width=True)

# Confusion matrix (best model)
best_pipe = pipes[best_name]
y_pred = best_pipe.predict(X_test)
cm_df = churn.confusion_df(y_test, y_pred)
st.markdown("#### 🧩 Confusion Matrix (best model)")
cm1, cm2 = st.columns([1, 1])
with cm1:
    st.dataframe(cm_df, use_container_width=True)
with cm2:
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
    ax2.imshow(cm_df.values, cmap="Purples")
    cut = cm_df.values.max() / 2
    for i in range(2):
        for j in range(2):
            v = cm_df.values[i, j]
            ax2.text(j, i, v, ha="center", va="center",
                     color="white" if v > cut else "#1e1b4b",
                     fontsize=14, fontweight="bold")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(cm_df.columns, color="white", fontsize=8)
    ax2.set_yticklabels(cm_df.index, color="white", fontsize=8)
    ax2.set_title("Confusion Matrix", color="white")
    style_mpl(fig2, ax2, grid=False)
    st.pyplot(fig2)
    cm_b64 = fig_to_base64(fig2)

# ---------------- Predict ----------------
section(5, "Predict Churn", "Customer ki details dein aur churn ka khatra dekhein.")
defaults = churn.feature_defaults(df)
options = churn.category_options(df)
X_all, _, num, cat = churn.get_xy(df)
with st.form("churn_predict"):
    cols = st.columns(3)
    inputs = {}
    for i, c in enumerate(num):
        with cols[i % 3]:
            inputs[c] = st.number_input(c, value=float(defaults[c]))
    for i, c in enumerate(cat):
        with cols[i % 3]:
            opts = options[c]
            idx = opts.index(defaults[c]) if defaults[c] in opts else 0
            inputs[c] = st.selectbox(c, opts, index=idx)
    submitted = st.form_submit_button("Predict Churn ⚠️", type="primary")

if submitted:
    row = pd.DataFrame([{c: inputs[c] for c in num + cat}])
    proba = float(best_pipe.predict_proba(row)[0][1])
    risky = proba >= 0.5
    result_card(
        "⚠️ Churn Probability", f"{proba:.1%}",
        "⚠️ Likely to CHURN" if risky else "✅ Likely to STAY",
        pct=proba * 100, tone="risk" if risky else "safe",
    )
    st.toast(f"Churn probability: {proba:.1%}", icon="👥")

# ---------------- Report ----------------
section(6, "Download Report", "Poori analysis ki HTML report ek click mein.")
metrics = {r["Model"]: f"F1={r['F1']}, Acc={r['Accuracy']}, AUC={r['ROC-AUC']}"
           for _, r in results.iterrows()}
metrics["Best model"] = best_name
metrics["Churn rate in data"] = f"{rate:.1f}%"
html = build_html_report(
    "Customer Churn Prediction",
    "Classification models trained on telecom customer data to flag churn risk.",
    metrics, [mpl_b64, cm_b64],
    "Predictions support retention work — they are not automatic decisions about customers.",
)
st.download_button("📥 Download Churn Report", html,
                   file_name="churn_report.html", mime="text/html",
                   use_container_width=True)

render_footer()