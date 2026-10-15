#pages/3_HR_Analytics.py
"""Room 3 page — HR Analytics & Employee Attrition Prediction."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src import hr
from src.utils import (apply_custom_css, build_html_report, fig_to_base64,
                       play_voice, render_footer, render_sidebar, show_robot,
                       validate_upload)

st.set_page_config(page_title="HR Analytics — TriMind AI", page_icon="🧑‍💼",
                   layout="wide")
apply_custom_css()
render_sidebar()

st.markdown('<div class="hero-title" style="font-size:2.2rem">🧑‍💼 HR Analytics & Employee Attrition</div>',
            unsafe_allow_html=True)
st.caption("Classification room — predict who might leave, and why.")
show_robot(height=120)

# ---------------- Data source ----------------
src_choice = st.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True)
df = None
if src_choice == "Sample data":
    play_voice("loading")
    with st.status("Loading sample HR data…", expanded=True) as s:
        df = hr.load_data()
        s.update(label=f"Loaded {len(df):,} rows", state="complete")
else:
    up = st.file_uploader("Upload an HR CSV", type=["csv"])
    if up:
        raw = pd.read_csv(up)
        ok, msg = validate_upload(raw, hr.REQUIRED_COLS)
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
    df = hr.clean_data(df)
    s.update(label=f"Cleaned: {df.shape[0]:,} rows × {df.shape[1]} columns",
             state="complete")
st.toast("Data cleaned ✅", icon="🧹")

# ---------------- EDA ----------------
play_voice("analyzing")
st.subheader("🔍 Exploratory Analysis")
rate = df[hr.TARGET].mean() * 100
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Employees</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><div class="kpi-value">{rate:.1f}%</div><div class="kpi-label">Attrition Rate</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["Age"].mean():.1f}</div><div class="kpi-label">Avg Age</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-value">${df["MonthlyIncome"].mean():,.0f}</div><div class="kpi-label">Avg Monthly Income</div></div>', unsafe_allow_html=True)

e1, e2 = st.columns(2)
with e1:
    counts = df[hr.TARGET].map({1: "Left", 0: "Stayed"}).value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    st.plotly_chart(px.pie(counts, names="Status", values="Count",
                           title="Attrition Distribution", hole=0.4),
                    use_container_width=True)
with e2:
    st.plotly_chart(px.histogram(df, x="MonthlyIncome",
                                 color=df[hr.TARGET].map({1: "Left", 0: "Stayed"}),
                                 barmode="overlay", opacity=0.7,
                                 title="Monthly Income vs Attrition"),
                    use_container_width=True)

e3, e4 = st.columns(2)
with e3:
    if "Department" in df.columns:
        dt = df.groupby(["Department", hr.TARGET]).size().reset_index(name="n")
        dt["Status"] = dt[hr.TARGET].map({1: "Left", 0: "Stayed"})
        st.plotly_chart(px.bar(dt, x="Department", y="n", color="Status",
                               barmode="group", title="Attrition by Department"),
                        use_container_width=True)
with e4:
    if "OverTime" in df.columns:
        ot = df.groupby(["OverTime", hr.TARGET]).size().reset_index(name="n")
        ot["Status"] = ot[hr.TARGET].map({1: "Left", 0: "Stayed"})
        st.plotly_chart(px.bar(ot, x="OverTime", y="n", color="Status",
                               barmode="group", title="Attrition by Overtime"),
                        use_container_width=True)

# Matplotlib: satisfaction heatmap-style bar
sat = hr.satisfaction_by_attrition(df)
if sat is not None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = range(len(sat.index))
    w = 0.35
    ax.bar([i - w / 2 for i in x], sat[0], width=w, label="Stayed", color="#60a5fa")
    ax.bar([i + w / 2 for i in x], sat[1], width=w, label="Left", color="#a78bfa")
    ax.set_xticks(list(x)); ax.set_xticklabels(sat.index, rotation=30, ha="right", color="white")
    ax.set_title("Avg Satisfaction Scores: Stayed vs Left", color="white")
    ax.tick_params(colors="white"); ax.legend()
    fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#0E1117")
    st.pyplot(fig)
    mpl_b64 = fig_to_base64(fig)
else:
    mpl_b64 = ""

# ---------------- Train ----------------
play_voice("training")
st.subheader("🤖 Train Models")
if st.button("Train 3 Models", type="primary"):
    X_train, X_test, y_train, y_test, num, cat = hr.split(df)
    prog = st.progress(0, text="Training…")
    with st.status("Training models…", expanded=True) as s:
        pipes = {}
        for i, (name, model) in enumerate(hr.get_models().items()):
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
    results, probas = hr.evaluate(pipes, X_test, y_test)
    st.session_state["hr_pipes"] = pipes
    st.session_state["hr_results"] = results
    st.session_state["hr_splits"] = (X_test, y_test)
    st.toast("Training complete ✅", icon="🤖")
    play_voice("complete")
    st.rerun()

if "hr_results" not in st.session_state:
    st.info("Press **Train 3 Models** to compare classifiers.")
    render_footer()
    st.stop()

results = st.session_state["hr_results"]
pipes = st.session_state["hr_pipes"]
X_test, y_test = st.session_state["hr_splits"]

st.subheader("🏆 Model Comparison")
st.dataframe(results, use_container_width=True)
best_name = results.iloc[0]["Model"]
st.success(f"Best model: **{best_name}** (F1 = {results.iloc[0]['F1']})")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(results, x="Model", y="F1", color="Model",
                           title="F1-Score Comparison", text="F1"),
                    use_container_width=True)
with c2:
    melt = results.melt(id_vars="Model", value_vars=["Precision", "Recall"],
                        var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(melt, x="Model", y="Score", color="Metric",
                           barmode="group", title="Precision vs Recall"),
                    use_container_width=True)

# Confusion matrix (best model)
best_pipe = pipes[best_name]
y_pred = best_pipe.predict(X_test)
cm_df = hr.confusion_df(y_test, y_pred)
st.subheader("🧩 Confusion Matrix (best model)")
st.dataframe(cm_df, use_container_width=True)
fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
ax2.imshow(cm_df.values, cmap="Purples")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm_df.values[i, j], ha="center", va="center",
                 color="white", fontsize=14, fontweight="bold")
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(cm_df.columns, color="white", fontsize=8)
ax2.set_yticklabels(cm_df.index, color="white", fontsize=8)
ax2.set_title("Confusion Matrix", color="white")
fig2.patch.set_facecolor("#0E1117"); ax2.set_facecolor("#0E1117")
st.pyplot(fig2)
cm_b64 = fig_to_base64(fig2)

# ---------------- Predict ----------------
st.subheader("🔮 Predict Attrition")
defaults = hr.feature_defaults(df)
options = hr.category_options(df)
X_all, _, num, cat = hr.get_xy(df)
with st.form("hr_predict"):
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
    submitted = st.form_submit_button("Predict Attrition 🧑‍💼", type="primary")

if submitted:
    row = pd.DataFrame([{c: inputs[c] for c in num + cat}])
    proba = float(best_pipe.predict_proba(row)[0][1])
    label = "⚠️ Likely to LEAVE" if proba >= 0.5 else "✅ Likely to STAY"
    st.metric("Attrition Probability", f"{proba:.1%}", label)
    st.toast(f"Attrition probability: {proba:.1%}", icon="🧑‍💼")

# ---------------- Report ----------------
st.subheader("📥 Download Report")
metrics = {r["Model"]: f"F1={r['F1']}, Acc={r['Accuracy']}, AUC={r['ROC-AUC']}"
           for _, r in results.iterrows()}
metrics["Best model"] = best_name
metrics["Attrition rate in data"] = f"{rate:.1f}%"
figs = [b for b in [mpl_b64, cm_b64] if b]
html = build_html_report(
    "HR Analytics & Employee Attrition",
    "Classification models trained on HR data to flag attrition risk early.",
    metrics, figs,
    "Predictions support retention planning — they are not automatic employment decisions.",
)
st.download_button("Download HR Report", html,
                   file_name="hr_report.html", mime="text/html",
                   use_container_width=True)

render_footer()