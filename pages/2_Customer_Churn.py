# pages/2_Customer_Churn.py
"""Room 2 page — Customer Churn Prediction."""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import churn
from src.pipeline_ui import render_pipeline_workflow  # <--- NAYA: Dynamic pipeline import kiya
from src.ui import (PALETTE, apply_page_css, best_banner, kpi, page_header,
                    result_card, section, style_fig, style_mpl)
from src.utils import (apply_custom_css, build_html_report, fig_to_base64,
                       play_voice, render_footer, show_robot,
                       validate_upload)

st.set_page_config(page_title="Customer Churn — TriMind AI", page_icon="👥", layout="wide")
apply_custom_css()
apply_page_css(accent_rgb="96,165,250")

# --- BACK BUTTON ---
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("⬅️ Back to Home", use_container_width=True):
        st.switch_page("app.py") 
# ------------------

# ---------------- Tabs ko 2 rows mein todo (5 + 4) ----------------
st.markdown("""
<style>
div[data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    overflow: visible !important;
    gap: 6px 4px !important;
    row-gap: 8px !important;
}
div[data-baseweb="tab-list"] button[role="tab"] {
    flex: 0 0 calc(20% - 4px) !important;   /* 5 tabs per row */
    justify-content: center !important;
    border-radius: 10px !important;
    border: 1px solid rgba(96,165,250,0.25) !important;
    padding: 8px 6px !important;
    white-space: nowrap !important;
}
div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
    background: rgba(96,165,250,0.15) !important;
    border: 1px solid #60a5fa !important;
}
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

STATUS_COLORS = {"Churned": "#f472b6", "Stayed": "#60a5fa"}
MODEL_COLORS = ["#60a5fa", "#f472b6", "#fbbf24", "#34d399"]

# Safe defaults (NameError se bachne ke liye)
mpl_b64 = ""
cm_b64 = ""

# ---------------- Header ----------------
hc1, hc2 = st.columns([4, 1], vertical_alignment="center")
with hc1:
    page_header("👥", "Customer Churn Prediction", "Classification room — spot customers likely to leave, before they do.",
        badges=["Logistic Regression", "Random Forest", "Neural Network"], title_grad="linear-gradient(90deg,#93c5fd,#c4b5fd,#f0abfc)")
with hc2:
    show_robot(height=120)

# ---------------- Data Source ----------------
section(1, "Data Source", "Built-in sample data use karein ya apni CSV upload karein.")
src_choice = st.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True, label_visibility="collapsed")

# ==========================================
# YAHAN CHANGE HUA HAI: 
# Upload CSV select karne par naya dynamic pipeline chalega
# ==========================================
if src_choice == "Sample data":
    play_voice("loading")
    with st.status("Loading sample churn data…", expanded=True) as s:
        df = churn.load_data()
        s.update(label=f"Loaded {len(df):,} rows", state="complete")
        
    play_voice("cleaning")
    with st.status("Cleaning data…", expanded=True) as s:
        df = churn.clean_data(df)
        s.update(label=f"Cleaned: {df.shape[0]:,} rows × {df.shape[1]} columns", state="complete")
    st.toast("Data cleaned ✅", icon="🧹")
    rate = df[churn.TARGET].mean() * 100

else:
    up = st.file_uploader("Upload a CSV", type=["csv"])
    if up:
        # Sir ki hidayat ke mutabiq dynamic pipeline (Column delete, Encode, Scale)
        render_pipeline_workflow(up, key_prefix="churn_mlp", start_num=2)
    else:
        st.info("Upload a CSV file to start the dynamic pipeline.")
    
    # Sample data wala code na chale isliye yahin stop kar diya
    render_footer()
    st.stop()

# ==========================================
# Neeche ka code sirf Sample Data ke liye chalega
# ==========================================

# ---------------- Saved model auto-load ----------------
# FIX: @st.cache_resource lagaya gaya hai taake model baar baar load na ho
@st.cache_resource(show_spinner=False)
def _load_saved_churn_model():
    """Saved churn model wapas load karo (churn.load_model ho to wo, warna joblib)."""
    loader = getattr(churn, "load_model", None)
    if loader is not None:
        return loader("churn_best_model.pkl")
    path = Path(__file__).resolve().parent.parent / "models" / "churn_best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


if "churn_pipes" not in st.session_state:
    try:
        saved = _load_saved_churn_model()
        _, X_te, _, y_te, _, _ = churn.split(df)
        res, _ = churn.evaluate({"Saved Model": saved}, X_te, y_te)
        st.session_state["churn_pipes"] = {"Saved Model": saved}
        st.session_state["churn_results"] = res
        st.session_state["churn_splits"] = (X_te, y_te)
        st.session_state["churn_autoloaded"] = True
    except Exception:
        pass  # model file nahi mila ya purane columns se match nahi karta

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab_cmp, tab6, tab7, tab8 = st.tabs([
    "🏠 Home", "📊 Explorer", "📈 Dashboard", "🔍 Analysis",
    "🤖 Model Training", "🆚 Model Comparison", "🎚️ Threshold Analysis",
    "🔮 Predict Churn", "🛡️ Retention"
])

with tab1:
    st.header("Business Problem")
    st.write("A subscription-based company is losing customers. We need to identify which customers are likely to leave and why.")
    st.info("This platform uses classification models to predict churn risk and suggest retention actions.")

with tab2:
    st.header("Customer Explorer")
    contract_filter = st.multiselect("Contract Type", df['Contract'].unique() if 'Contract' in df.columns else [], default=df['Contract'].unique() if 'Contract' in df.columns else [])
    filtered_df = df[df['Contract'].isin(contract_filter)] if 'Contract' in df.columns else df
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False), "filtered_churn.csv", "text/csv")

with tab3:
    play_voice("analyzing")
    st.header("Churn Dashboard KPIs")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "👥", f"{len(df):,}", "Total Customers", "linear-gradient(90deg,#60a5fa,#a78bfa)")
    kpi(k2, "⚠️", f"{rate:.1f}%", "Churn Rate", "linear-gradient(90deg,#fb7185,#f472b6)")
    kpi(k3, "⏳", f'{df["tenure"].mean():.1f}', "Avg Tenure (months)", "linear-gradient(90deg,#34d399,#60a5fa)")
    kpi(k4, "💳", f'${df["MonthlyCharges"].mean():.2f}', "Avg Monthly Charges", "linear-gradient(90deg,#fbbf24,#f472b6)")

with tab4:
    status_series = df[churn.TARGET].map({1: "Churned", 0: "Stayed"})
    e1, e2 = st.columns(2)
    with e1:
        counts = status_series.value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        st.plotly_chart(style_fig(px.pie(counts, names="Status", values="Count", title="Churn Distribution", hole=0.45, color="Status", color_discrete_map=STATUS_COLORS)), use_container_width=True)
    with e2:
        st.plotly_chart(style_fig(px.histogram(df, x="tenure", color=status_series, barmode="overlay", opacity=0.7, title="Tenure vs Churn", color_discrete_map=STATUS_COLORS)), use_container_width=True)

    e3, e4 = st.columns(2)
    with e3:
        st.plotly_chart(style_fig(px.box(df, x=status_series, y="MonthlyCharges", color=status_series, title="Monthly Charges by Churn Status", color_discrete_map=STATUS_COLORS)), use_container_width=True)
    with e4:
        if "Contract" in df.columns:
            ct = df.groupby(["Contract", churn.TARGET]).size().reset_index(name="n")
            ct["Status"] = ct[churn.TARGET].map({1: "Churned", 0: "Stayed"})
            st.plotly_chart(style_fig(px.bar(ct, x="Contract", y="n", color="Status", barmode="group", title="Churn by Contract Type", color_discrete_map=STATUS_COLORS)), use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bucket_df = df.assign(tenure_bucket=pd.cut(df["tenure"], bins=[0, 12, 24, 48, 100], labels=["0-12", "13-24", "25-48", "49+"]))
    bucket_rate = bucket_df.groupby("tenure_bucket", observed=True)[churn.TARGET].mean() * 100
    ax.bar(bucket_rate.index.astype(str), bucket_rate.values, color=PALETTE[: len(bucket_rate)])
    ax.set_title("Churn Rate % by Tenure Bucket", color="white")
    style_mpl(fig, ax)
    st.pyplot(fig)
    mpl_b64 = fig_to_base64(fig)

with tab5:
    play_voice("training")
    st.header("Train & Compare Models")
    if st.button("🚀 Train 3 Models", type="primary"):
        X_train, X_test, y_train, y_test, num, cat = churn.split(df)
        prog = st.progress(0, text="Training…")
        with st.status("Training models…", expanded=True) as s:
            prep_template = ColumnTransformer([
                ("num", StandardScaler(), num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ])
            pipes, times = {}, {}
            models = churn.get_models()
            for i, (name, model) in enumerate(models.items()):
                st.write(f"Step {i + 1}/{len(models)} — {name}")
                pipe = Pipeline([("prep", clone(prep_template)), ("model", model)])
                t0 = time.perf_counter()
                pipe.fit(X_train, y_train)
                times[name] = round(time.perf_counter() - t0, 3)
                pipes[name] = pipe
                prog.progress((i + 1) / len(models), text=f"Trained {name}")
            s.update(label="All models trained ✅", state="complete")
        prog.empty()

        results, probas = churn.evaluate(pipes, X_test, y_test)

        # SAVE MODEL (Assignment Requirement)
        best_name = results.iloc[0]["Model"]
        churn.save_model(pipes[best_name], "churn_best_model.pkl")

        st.session_state["churn_pipes"] = pipes
        st.session_state["churn_results"] = results
        st.session_state["churn_splits"] = (X_test, y_test)
        st.session_state["churn_times"] = times
        st.session_state["churn_autoloaded"] = False
        st.toast("Training complete & model saved ✅", icon="🤖")
        play_voice("complete")
        st.rerun()

    if "churn_results" in st.session_state:
        results = st.session_state["churn_results"]
        pipes = st.session_state["churn_pipes"]
        X_test, y_test = st.session_state["churn_splits"]

        if st.session_state.get("churn_autoloaded"):
            st.info("Pehle se saved model load hua hai. Sab 3 models compare karne ke liye **Train 3 Models** dabayein.")

        st.dataframe(results, use_container_width=True)
        best_name = results.iloc[0]["Model"]
        best_banner(best_name, f"F1 = {results.iloc[0]['F1']}")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(style_fig(px.bar(results, x="Model", y="F1", color="Model", title="F1-Score Comparison", text="F1")), use_container_width=True)
        with c2:
            melt = results.melt(id_vars="Model", value_vars=["Precision", "Recall"], var_name="Metric", value_name="Score")
            st.plotly_chart(style_fig(px.bar(melt, x="Model", y="Score", color="Metric", barmode="group", title="Precision vs Recall")), use_container_width=True)

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
                    ax2.text(j, i, v, ha="center", va="center", color="white" if v > cut else "#1e1b4b", fontsize=14, fontweight="bold")
            ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
            ax2.set_xticklabels(cm_df.columns, color="white", fontsize=8)
            ax2.set_yticklabels(cm_df.index, color="white", fontsize=8)
            ax2.set_title("Confusion Matrix", color="white")
            style_mpl(fig2, ax2, grid=False)
            st.pyplot(fig2)
            cm_b64 = fig_to_base64(fig2)

        st.markdown("### Download Report")
        metrics = {r["Model"]: f"F1={r['F1']}, Acc={r['Accuracy']}, AUC={r['ROC-AUC']}" for _, r in results.iterrows()}
        metrics["Best model"] = best_name
        metrics["Churn rate in data"] = f"{rate:.1f}%"
        figs = [b for b in [mpl_b64, cm_b64] if b]
        html = build_html_report("Customer Churn Prediction", "Classification models trained on telecom customer data.", metrics, figs, "Predictions support retention work.")
        st.download_button("📥 Download Churn Report", html, file_name="churn_report.html", mime="text/html", use_container_width=True)
    else:
        st.info("Press **Train 3 Models** to compare classifiers.")

with tab_cmp:
    st.header("🆚 Model Comparison")
    st.caption("Teeno models ka metrics, ROC curve, errors aur training time ek jagah.")
    if "churn_results" not in st.session_state:
        st.warning("Train models first (🤖 Model Training tab).")
    else:
        cmp_res = st.session_state["churn_results"].copy()
        cmp_pipes = st.session_state["churn_pipes"]
        cmp_Xt, cmp_yt = st.session_state["churn_splits"]
        metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        for m in metric_cols:
            cmp_res[m] = pd.to_numeric(cmp_res[m], errors="coerce")
        color_map = {n: MODEL_COLORS[i % len(MODEL_COLORS)] for i, n in enumerate(cmp_res["Model"])}

        if len(cmp_res) < 2:
            st.info("Abhi sirf ek saved model load hai. Teeno ka muqabla karne ke liye **Model Training** tab mein **Train 3 Models** dabayein.")

        st.subheader("🏆 Har metric ka winner")
        wcols = st.columns(len(metric_cols))
        for col, m in zip(wcols, metric_cols):
            if cmp_res[m].notna().any():
                top = cmp_res.loc[cmp_res[m].idxmax()]
                col.metric(m, top["Model"], f"{top[m]:.3f}", delta_color="off")
            else:
                col.metric(m, "—")

        rank_s = (cmp_res.set_index("Model")[metric_cols]
                  .rank(ascending=False, method="min").mean(axis=1).sort_values())
        st.success(f"🥇 Overall best: **{rank_s.index[0]}** (average rank {rank_s.iloc[0]:.2f}, 1 = sab se behtar)")

        st.subheader("📋 Metrics table (best value highlighted)")
        styled = (cmp_res.style
                  .highlight_max(subset=metric_cols, props="background-color:#60a5fa;color:#111;font-weight:bold")
                  .format({m: "{:.4f}" for m in metric_cols}, na_rep="—"))
        st.dataframe(styled, use_container_width=True)
        st.download_button("📥 Download comparison (CSV)", cmp_res.to_csv(index=False),
                           file_name="churn_model_comparison.csv", mime="text/csv")

        g1, g2 = st.columns(2)
        with g1:
            melt_all = cmp_res.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            fig_b = px.bar(melt_all, x="Metric", y="Score", color="Model", barmode="group",
                           title="All Metrics Side by Side", color_discrete_map=color_map, range_y=[0, 1])
            st.plotly_chart(style_fig(fig_b), use_container_width=True)
        with g2:
            fig_r = go.Figure()
            for _, row in cmp_res.iterrows():
                vals = [0 if pd.isna(row[m]) else float(row[m]) for m in metric_cols]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals + vals[:1], theta=metric_cols + metric_cols[:1],
                    fill="toself", name=row["Model"], line_color=color_map[row["Model"]], opacity=0.6))
            fig_r.update_layout(title="Radar Chart", polar=dict(radialaxis=dict(range=[0, 1])))
            st.plotly_chart(style_fig(fig_r), use_container_width=True)

        st.subheader("📉 ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                     line=dict(dash="dash", color="#6b7280")))
        for name, pipe in cmp_pipes.items():
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(cmp_Xt)[:, 1]
                fpr, tpr, _ = roc_curve(cmp_yt, proba)
                auc_v = cmp_res.loc[cmp_res["Model"] == name, "ROC-AUC"].iloc[0]
                label = f"{name} (AUC={auc_v:.3f})" if pd.notna(auc_v) else name
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label,
                                             line=dict(width=3, color=color_map.get(name))))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_fig(fig_roc), use_container_width=True)

        st.subheader("🧩 Errors: kaun kitne churners miss karta hai?")
        err_rows, cms = [], {}
        for name, pipe in cmp_pipes.items():
            cm_d = churn.confusion_df(cmp_yt, pipe.predict(cmp_Xt))
            cms[name] = cm_d
            tn, fp, fn, tp = cm_d.values.ravel()
            err_rows.append({"Model": name, "Sahi pakre (TP)": int(tp), "Missed churners (FN)": int(fn),
                             "False alarms (FP)": int(fp), "Sahi stayed (TN)": int(tn)})
        st.dataframe(pd.DataFrame(err_rows), use_container_width=True)
        st.caption("Churn mein **Recall / FN** aham hai: FN kam ho to jaane wale customers kam miss hote hain.")

        ccols = st.columns(len(cms))
        for col, (name, cm_d) in zip(ccols, cms.items()):
            with col:
                fig_cm = px.imshow(cm_d, text_auto=True, color_continuous_scale="Purples", title=name)
                fig_cm.update_layout(coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig_cm), use_container_width=True)

        times = st.session_state.get("churn_times")
        if times:
            st.subheader("⏱️ Training time")
            t_df = pd.DataFrame({"Model": list(times.keys()), "Seconds": list(times.values())})
            st.plotly_chart(style_fig(px.bar(t_df, x="Model", y="Seconds", color="Model", text="Seconds",
                                             color_discrete_map=color_map, title="Training Time (seconds)")),
                            use_container_width=True)

        st.subheader("📝 Natija")
        best_f1 = cmp_res.loc[cmp_res["F1"].idxmax()]
        best_rec = cmp_res.loc[cmp_res["Recall"].idxmax()]
        best_pre = cmp_res.loc[cmp_res["Precision"].idxmax()]
        st.markdown(
            f"- **Balanced (F1):** {best_f1['Model']} ({best_f1['F1']:.3f})\n"
            f"- **Sab se zyada churners pakre (Recall):** {best_rec['Model']} ({best_rec['Recall']:.3f})\n"
            f"- **Kam false alarms (Precision):** {best_pre['Model']} ({best_pre['Precision']:.3f})"
        )

with tab6:
    st.header("🎚️ Threshold Analysis")
    if "churn_pipes" in st.session_state:
        best_pipe = st.session_state["churn_pipes"][st.session_state["churn_results"].iloc[0]["Model"]]
        X_test, y_test = st.session_state["churn_splits"]
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        threshold = st.slider("Select Probability Threshold (Default = 0.5)", 0.0, 1.0, 0.5, 0.05)
        y_pred = (y_proba >= threshold).astype(int)

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2f}")
        c2.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2f}")
        c3.metric("F1-Score", f"{f1_score(y_test, y_pred, zero_division=0):.2f}")
        st.info("Lowering the threshold increases Recall (catches more churners) but reduces Precision.")
    else:
        st.warning("Train models in 'Model Training' tab first.")

with tab7:
    if "churn_pipes" in st.session_state:
        best_pipe = st.session_state["churn_pipes"][st.session_state["churn_results"].iloc[0]["Model"]]
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
                    inputs[c] = st.selectbox(c, opts, index=opts.index(defaults[c]) if defaults[c] in opts else 0)
            submitted = st.form_submit_button("Predict Churn ⚠️", type="primary")

        if submitted:
            row = pd.DataFrame([{c: inputs[c] for c in num + cat}])
            proba = float(best_pipe.predict_proba(row)[0][1])
            risky = proba >= 0.5
            result_card("⚠️ Churn Probability", f"{proba:.1%}", "⚠️ Likely to CHURN" if risky else "✅ Likely to STAY", pct=proba * 100, tone="risk" if risky else "safe")
            st.toast(f"Churn probability: {proba:.1%}", icon="👥")
    else:
        st.warning("Train models first.")

with tab8:
    st.header("🛡️ Retention Dashboard")
    if "churn_pipes" in st.session_state:
        best_pipe = st.session_state["churn_pipes"][st.session_state["churn_results"].iloc[0]["Model"]]
        X_all, _, _, _ = churn.get_xy(df)
        scored = df.copy()
        scored["Churn_Probability"] = best_pipe.predict_proba(X_all)[:, 1]

        high_risk = scored[scored["Churn_Probability"] >= 0.7].copy()
        st.metric("High Risk Customers (>=70% probability)", len(high_risk))

        if not high_risk.empty:
            st.dataframe(high_risk[["tenure", "MonthlyCharges", "Churn_Probability"]].sort_values(by="Churn_Probability", ascending=False), use_container_width=True)
            st.download_button("📥 Download High Risk List", high_risk.to_csv(index=False), "high_risk_churn.csv", "text/csv")
    else:
        st.warning("Train models first.")

render_footer()