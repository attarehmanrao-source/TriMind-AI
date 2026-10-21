# pages/3_HR_Analytics.py
"""Room 3 page — HR Analytics & Employee Attrition Prediction."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.metrics import roc_curve
import streamlit as st

from src import hr
from src.ui import (PALETTE, apply_page_css, best_banner, kpi, page_header,
                    result_card, section, style_fig, style_mpl)
from src.utils import (apply_custom_css, build_html_report, fig_to_base64,
                       play_voice, render_footer, render_sidebar, show_robot,
                       validate_upload)

st.set_page_config(page_title="HR Analytics — TriMind AI", page_icon="🧑‍💼", layout="wide")
apply_custom_css()
render_sidebar()
apply_page_css(accent_rgb="251,191,36")

# ---------------- Tabs ko 2 rows mein todo (6 + 5) ----------------
st.markdown("""
<style>
div[data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    overflow: visible !important;
    gap: 6px 4px !important;
    row-gap: 8px !important;
}
div[data-baseweb="tab-list"] button[role="tab"] {
    flex: 0 0 calc(16.66% - 4px) !important;   /* 6 tabs per row */
    justify-content: center !important;
    border-radius: 10px !important;
    border: 1px solid rgba(251,191,36,0.25) !important;
    padding: 8px 6px !important;
    white-space: nowrap !important;
}
div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
    background: rgba(251,191,36,0.15) !important;
    border: 1px solid #fbbf24 !important;
}
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

STATUS_COLORS = {"Left": "#fbbf24", "Stayed": "#60a5fa"}
MODEL_COLORS = ["#fbbf24", "#60a5fa", "#f472b6", "#34d399"]

# FIX 1: ye variables upar define hain taake NameError kabhi na aaye
mpl_b64 = ""
cm_b64 = ""

# ---------------- Header ----------------
hc1, hc2 = st.columns([4, 1], vertical_alignment="center")
with hc1:
    page_header("🧑‍💼", "HR Analytics & Employee Attrition", "Classification room — predict who might leave, and why.",
        badges=["Logistic Regression", "Random Forest", "Neural Network"], title_grad="linear-gradient(90deg,#fde68a,#fbbf24,#f472b6)")
with hc2:
    show_robot(height=120)

# ---------------- Data Source ----------------
section(1, "Data Source", "Built-in sample data use karein ya apni CSV upload karein.")
src_choice = st.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True, label_visibility="collapsed")
df = None
if src_choice == "Sample data":
    play_voice("loading")
    with st.status("Loading HR data…", expanded=True) as s:
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
    s.update(label=f"Cleaned: {df.shape[0]:,} rows × {df.shape[1]} columns", state="complete")
st.toast("Data cleaned ✅", icon="🧹")

rate = df[hr.TARGET].mean() * 100

# ---------------- FIX 2: Saved model auto-load ----------------
# Refresh (F5) ke baad session_state khali hota hai. Agar models/hr_best_model.pkl
# maujood hai to usay wapas load kar lo, taake "Train first" na aaye.
if "hr_pipes" not in st.session_state:
    try:
        saved = hr.load_model("hr_best_model.pkl")
        _, X_te, _, y_te, _, _ = hr.split(df)
        res, _ = hr.evaluate({"Saved Model": saved}, X_te, y_te)
        st.session_state["hr_pipes"] = {"Saved Model": saved}
        st.session_state["hr_results"] = res
        st.session_state["hr_splits"] = (X_te, y_te)
        st.session_state["hr_autoloaded"] = True
    except FileNotFoundError:
        pass  # pehli baar: user ko train karna hi parega
    except Exception:
        pass  # purana model naye data ke columns se match nahi karta

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab_cmp, tab10, tab11 = st.tabs([
    "🏠 Home", "🩺 Data Quality", "📈 Dashboard", "👥 Demographics",
    "💰 Compensation", "🧠 Engagement", "🚪 Attrition Analysis",
    "⚙️ Preprocessing Lab", "🤖 Model Evaluation", "🆚 Model Comparison", "🔮 Risk Assessment", "⚖️ Responsible ML"
])

with tab1:
    st.header("Business Problem & Ethics")
    st.write("An organization is experiencing employee attrition. The HR department wants to understand which employees are likely to leave, and how factors like overtime, salary, and work-life balance affect retention.")
    st.error("**Ethical Requirement:** This system must not be presented as a tool for firing or punishing employees. It is only for voluntary retention support.")

with tab2:
    st.header("Data Quality Report")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values:**")
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0], use_container_width=True)
        if df.isnull().sum().sum() == 0:
            st.success("No missing values!")
    with col2:
        st.write("**Data Types:**")
        st.dataframe(df.dtypes.astype(str), use_container_width=True)
    st.write(f"**Total Duplicates:** {df.duplicated().sum()}")

with tab3:
    play_voice("analyzing")
    st.header("HR Executive Dashboard")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "🧑‍💼", f"{len(df):,}", "Total Employees", "linear-gradient(90deg,#fbbf24,#f472b6)")
    kpi(k2, "🚪", f"{rate:.1f}%", "Attrition Rate", "linear-gradient(90deg,#fb7185,#f472b6)")
    kpi(k3, "🎂", f'{df["Age"].mean():.1f}', "Avg Age", "linear-gradient(90deg,#60a5fa,#a78bfa)")
    kpi(k4, "💵", f'${df["MonthlyIncome"].mean():,.0f}', "Avg Monthly Income", "linear-gradient(90deg,#34d399,#60a5fa)")

with tab4:
    st.header("Workforce Demographics")
    status_series = df[hr.TARGET].map({1: "Left", 0: "Stayed"})
    e1, e2 = st.columns(2)
    with e1:
        if "Department" in df.columns:
            st.plotly_chart(style_fig(px.histogram(df, x="Department", color=status_series, barmode="group", title="Attrition by Department", color_discrete_map=STATUS_COLORS)), use_container_width=True)
    with e2:
        if "Gender" in df.columns:
            st.plotly_chart(style_fig(px.pie(df, names="Gender", title="Gender Distribution", hole=0.4)), use_container_width=True)

# status_series tab4 ke andar bana tha; tab5/6 ke liye yahan bhi safe define
status_series = df[hr.TARGET].map({1: "Left", 0: "Stayed"})

with tab5:
    st.header("Compensation Analysis")
    st.plotly_chart(style_fig(px.box(df, x="Department" if "Department" in df.columns else status_series, y="MonthlyIncome", color=status_series, title="Income by Department & Attrition", color_discrete_map=STATUS_COLORS)), use_container_width=True)

with tab6:
    st.header("Employee Engagement")
    st.plotly_chart(style_fig(px.box(df, x=status_series, y="JobSatisfaction", color=status_series, title="Job Satisfaction vs Attrition", color_discrete_map=STATUS_COLORS)), use_container_width=True)
    if "OverTime" in df.columns:
        st.plotly_chart(style_fig(px.histogram(df, x="OverTime", color=status_series, barmode="group", title="Overtime vs Attrition", color_discrete_map=STATUS_COLORS)), use_container_width=True)

    sat = hr.satisfaction_by_attrition(df)
    if sat is not None:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = range(len(sat.index))
        w = 0.35
        ax.bar([i - w / 2 for i in x], sat[0], width=w, label="Stayed", color="#60a5fa")
        ax.bar([i + w / 2 for i in x], sat[1], width=w, label="Left", color="#fbbf24")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sat.index, rotation=30, ha="right", color="white")
        ax.set_title("Avg Satisfaction Scores: Stayed vs Left", color="white")
        ax.legend(facecolor="#161A27", edgecolor="#3b3f63", labelcolor="white")
        style_mpl(fig, ax)
        st.pyplot(fig)
        mpl_b64 = fig_to_base64(fig)

with tab7:
    st.header("Attrition Analysis")
    st.write("Deep dive into why employees are leaving (based on filters from Demographics tab).")

with tab8:
    st.header("Preprocessing Laboratory")
    X, y, num, cat = hr.get_xy(df)
    st.write("**Numerical Features:**", num)
    st.write("**Categorical Features:**", cat)
    st.write(f"**Processed Feature Count (After One-Hot Encoding):** {len(pd.get_dummies(X, drop_first=True).columns)}")
    st.write(f"**Training/Testing Split shape:** {int(len(df)*0.8)} rows Train, {int(len(df)*0.2)} rows Test")

with tab9:
    play_voice("training")
    st.header("Train & Compare Models")

    if st.button("🚀 Train 3 Models", type="primary"):
        X_train, X_test, y_train, y_test, num, cat = hr.split(df)
        prog = st.progress(0, text="Training…")
        with st.status("Training models…", expanded=True) as s:
            # FIX 3: duplicate code hata diya, hr.train_all() use karo
            # (har model ka apna alag preprocessor clone hota hai)
            from sklearn.base import clone
            from sklearn.pipeline import Pipeline
            prep_template = hr._build_preprocessor(num, cat)
            pipes = {}
            times = {}
            models = hr.get_models()
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

        results, probas = hr.evaluate(pipes, X_test, y_test)

        # SAVE MODEL (Assignment Requirement)
        best_name = results.iloc[0]["Model"]
        hr.save_model(pipes[best_name], "hr_best_model.pkl", metadata=hr.build_metadata(df))

        st.session_state["hr_pipes"] = pipes
        st.session_state["hr_results"] = results
        st.session_state["hr_splits"] = (X_test, y_test)
        st.session_state["hr_autoloaded"] = False
        st.session_state["hr_times"] = times
        st.toast("Training complete & model saved ✅", icon="🤖")
        play_voice("complete")
        st.rerun()

    if "hr_results" in st.session_state:
        results = st.session_state["hr_results"]
        pipes = st.session_state["hr_pipes"]
        X_test, y_test = st.session_state["hr_splits"]

        if st.session_state.get("hr_autoloaded"):
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
        cm_df = hr.confusion_df(y_test, y_pred)

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
        metrics["Attrition rate in data"] = f"{rate:.1f}%"
        figs = [b for b in [mpl_b64, cm_b64] if b]
        html = build_html_report("HR Analytics & Employee Attrition", "Classification models trained on HR data.", metrics, figs, "Predictions support retention planning.")
        st.download_button("📥 Download HR Report", html, file_name="hr_report.html", mime="text/html", use_container_width=True)
    else:
        st.info("Press **Train 3 Models** to compare classifiers.")

with tab_cmp:
    st.header("🆚 Model Comparison")
    st.caption("Teeno models ka metrics, ROC curve, errors aur training time ek jagah.")
    if "hr_results" not in st.session_state:
        st.warning("Train models first.")
    else:
        cmp_res = st.session_state["hr_results"].copy()
        cmp_pipes = st.session_state["hr_pipes"]
        cmp_Xt, cmp_yt = st.session_state["hr_splits"]
        metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        for m in metric_cols:
            cmp_res[m] = pd.to_numeric(cmp_res[m], errors="coerce")
        color_map = {n: MODEL_COLORS[i % len(MODEL_COLORS)] for i, n in enumerate(cmp_res["Model"])}

        if len(cmp_res) < 2:
            st.info("Abhi sirf ek saved model load hai. Teeno ka muqabla karne ke liye **Model Evaluation** tab mein **Train 3 Models** dabayein.")

        # ---- Winners ----
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

        # ---- Table ----
        st.subheader("📋 Metrics table (best value highlighted)")
        styled = (cmp_res.style
                  .highlight_max(subset=metric_cols, props="background-color:#fbbf24;color:#111;font-weight:bold")
                  .format({m: "{:.4f}" for m in metric_cols}, na_rep="—"))
        st.dataframe(styled, use_container_width=True)
        st.download_button("📥 Download comparison (CSV)", cmp_res.to_csv(index=False),
                           file_name="hr_model_comparison.csv", mime="text/csv")

        # ---- Bar + Radar ----
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

        # ---- ROC curves ----
        st.subheader("📉 ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                     line=dict(dash="dash", color="#6b7280")))
        for name, pipe in cmp_pipes.items():
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(cmp_Xt)[:, 1]
                fpr, tpr, _ = roc_curve(cmp_yt, proba)
                auc_v = cmp_res.loc[cmp_res["Model"] == name, "ROC-AUC"].iloc[0]
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_v:.3f})",
                                             line=dict(width=3, color=color_map.get(name))))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_fig(fig_roc), use_container_width=True)

        # ---- Errors ----
        st.subheader("🧩 Errors: kaun kitne employees miss karta hai?")
        err_rows, cms = [], {}
        for name, pipe in cmp_pipes.items():
            cm_d = hr.confusion_df(cmp_yt, pipe.predict(cmp_Xt))
            cms[name] = cm_d
            tn, fp, fn, tp = cm_d.values.ravel()
            err_rows.append({"Model": name, "Sahi pakre (TP)": int(tp), "Missed leavers (FN)": int(fn),
                             "False alarms (FP)": int(fp), "Sahi stayed (TN)": int(tn)})
        st.dataframe(pd.DataFrame(err_rows), use_container_width=True)
        st.caption("HR mein **Recall / FN** aham hai: FN kam ho to chhodne wale employees kam miss hote hain.")

        ccols = st.columns(len(cms))
        for col, (name, cm_d) in zip(ccols, cms.items()):
            with col:
                fig_cm = px.imshow(cm_d, text_auto=True, color_continuous_scale="Purples", title=name)
                fig_cm.update_layout(coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig_cm), use_container_width=True)

        # ---- Training time ----
        times = st.session_state.get("hr_times")
        if times:
            st.subheader("⏱️ Training time")
            t_df = pd.DataFrame({"Model": list(times.keys()), "Seconds": list(times.values())})
            st.plotly_chart(style_fig(px.bar(t_df, x="Model", y="Seconds", color="Model", text="Seconds",
                                             color_discrete_map=color_map, title="Training Time (seconds)")),
                            use_container_width=True)

        # ---- Verdict ----
        st.subheader("📝 Natija")
        best_f1 = cmp_res.loc[cmp_res["F1"].idxmax()]
        best_rec = cmp_res.loc[cmp_res["Recall"].idxmax()]
        best_pre = cmp_res.loc[cmp_res["Precision"].idxmax()]
        st.markdown(
            f"- **Balanced (F1):** {best_f1['Model']} ({best_f1['F1']:.3f})\n"
            f"- **Sab se zyada leavers pakre (Recall):** {best_rec['Model']} ({best_rec['Recall']:.3f})\n"
            f"- **Kam false alarms (Precision):** {best_pre['Model']} ({best_pre['Precision']:.3f})"
        )

with tab10:
    if "hr_pipes" in st.session_state:
        best_pipe = st.session_state["hr_pipes"][st.session_state["hr_results"].iloc[0]["Model"]]
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
                    inputs[c] = st.selectbox(c, opts, index=opts.index(defaults[c]) if defaults[c] in opts else 0)
            submitted = st.form_submit_button("Predict Attrition 🧑‍💼", type="primary")

        if submitted:
            row = pd.DataFrame([{c: inputs[c] for c in num + cat}])
            proba = float(best_pipe.predict_proba(row)[0][1])
            risky = proba >= 0.5
            result_card("🧑‍💼 Attrition Probability", f"{proba:.1%}", "⚠️ Likely to LEAVE" if risky else "✅ Likely to STAY", pct=proba * 100, tone="risk" if risky else "safe")
            st.toast(f"Attrition probability: {proba:.1%}", icon="🧑‍💼")
    else:
        st.warning("Train models first.")

with tab11:
    st.header("⚖️ Responsible ML Review")
    st.warning("This prediction is intended for workforce-level analysis and voluntary retention support. It must not be used as the sole basis for termination, promotion, salary, disciplinary, or hiring decisions.")
    if "hr_pipes" in st.session_state:
        X_test, y_test = st.session_state["hr_splits"]
        if "Gender" in X_test.columns:
            best_pipe = st.session_state["hr_pipes"][st.session_state["hr_results"].iloc[0]["Model"]]
            preds = best_pipe.predict(X_test)

            eval_df = X_test.copy()
            eval_df["Actual"] = y_test
            eval_df["Predicted"] = preds

            group_metrics = eval_df.groupby("Gender").agg(
                Group_Size=("Actual", "count"),
                Actual_Attrition=("Actual", "mean"),
                Predicted_Attrition=("Predicted", "mean")
            )
            st.dataframe(group_metrics, use_container_width=True)
            st.write("Compare actual vs predicted attrition rates across groups to ensure fairness.")
    else:
        st.warning("Train models first.")

render_footer()