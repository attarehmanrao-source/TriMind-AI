# pages/1_Sales_Analytics.py
"""Room 1 page — Sales Analytics & Profit Prediction."""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

from src import sales
from src.utils import (apply_custom_css, build_html_report, fig_to_base64,
                       play_voice, render_footer, render_sidebar, show_robot,
                       validate_upload)
from src.ui import (PALETTE, apply_page_css, best_banner, kpi, page_header,
                    section, style_fig, style_mpl, result_card)

st.set_page_config(page_title="Sales Analytics — TriMind AI", page_icon="📈", layout="wide")
apply_custom_css()
render_sidebar()
apply_page_css(accent_rgb="52,211,153")

# ---------------- Tabs ko 2 rows mein todo (4 + 4) ----------------
st.markdown("""
<style>
div[data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    overflow: visible !important;
    gap: 6px 4px !important;
    row-gap: 8px !important;
}
div[data-baseweb="tab-list"] button[role="tab"] {
    flex: 0 0 calc(25% - 4px) !important;   /* 4 tabs per row */
    justify-content: center !important;
    border-radius: 10px !important;
    border: 1px solid rgba(52,211,153,0.25) !important;
    padding: 8px 6px !important;
    white-space: nowrap !important;
}
div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
    background: rgba(52,211,153,0.15) !important;
    border: 1px solid #34d399 !important;
}
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

PALETTE = ["#8b5cf6", "#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#fb923c"]
MODEL_COLORS = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24"]

# Safe defaults (NameError se bachne ke liye)
mpl_b64 = ""
avp_b64 = ""

# ---------------- Header ----------------
hc1, hc2 = st.columns([4, 1], vertical_alignment="center")
with hc1:
    page_header(
        "📈", "Sales Analytics & Profit Prediction",
        "Regression room — forecast profit from orders, discounts and segments.",
        badges=["Linear Regression", "Random Forest", "Neural Network"],
        title_grad="linear-gradient(90deg,#6ee7b7,#60a5fa,#c4b5fd)",
    )
with hc2:
    show_robot(height=120)

# ---------------- Data Source ----------------
section(1, "Data Source", "Built-in sample data use karein ya apni CSV upload karein.")
src_choice = st.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True, label_visibility="collapsed")
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
    s.update(label=f"Cleaned: {df.shape[0]:,} rows × {df.shape[1]} columns", state="complete")
st.toast("Data cleaned ✅", icon="🧹")


# ---------------- Saved model auto-load ----------------
def _load_saved_sales_model():
    """Saved sales model wapas load karo (sales.load_model ho to wo, warna joblib)."""
    loader = getattr(sales, "load_model", None)
    if loader is not None:
        return loader("sales_best_model.pkl")
    path = Path(__file__).resolve().parent.parent / "models" / "sales_best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


if "sales_pipes" not in st.session_state:
    try:
        saved = _load_saved_sales_model()
        X_tr, X_te, y_tr, y_te = sales.split(df)
        res = sales.evaluate({"Saved Model": saved}, X_te, y_te)
        # Agar saved model ka R² NaN aaye (purana/ghalat model), to usay use na karein
        if pd.to_numeric(res["R²"], errors="coerce").notna().any():
            st.session_state["sales_pipes"] = {"Saved Model": saved}
            st.session_state["sales_results"] = res
            st.session_state["sales_splits"] = (X_tr, X_te, y_tr, y_te)
            st.session_state["sales_autoloaded"] = True
    except Exception:
        pass  # model file nahi mila ya purane columns se match nahi karta

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab_cmp, tab7 = st.tabs([
    "🏠 Home", "📊 Dataset Explorer", "📈 Sales Dashboard",
    "🌍 Product & Regional", "🚚 Customer & Delivery", "🤖 Model Evaluation",
    "🆚 Model Comparison", "🔮 Predict Profit"
])

with tab1:
    st.header("Business Problem & Objectives")
    st.write("A retail company wants to understand its sales performance and improve profitability. The company wants to identify the most profitable products, categories, and regions, and predict the expected profit from future orders.")
    st.write("**Objective:** Build an interactive dashboard and train a regression model to predict profit.")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.header("Dataset Explorer")
    st.write("Filter and download data.")
    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
    with col2:
        cat_filter = st.multiselect("Category", df['Category'].unique(), default=df['Category'].unique())

    filtered_df = df[(df['Region'].isin(region_filter)) & (df['Category'].isin(cat_filter))]
    st.write(f"Showing {len(filtered_df):,} rows")
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False), "filtered_sales.csv", "text/csv")

with tab3:
    play_voice("analyzing")
    st.header("Key Performance Indicators (KPIs)")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "💵", f'${df["Sales"].sum():,.0f}', "Total Sales", "linear-gradient(90deg,#34d399,#60a5fa)")
    kpi(k2, "💰", f'${df["Profit"].sum():,.0f}', "Total Profit", "linear-gradient(90deg,#a78bfa,#f0abfc)")
    kpi(k3, "🧾", f'{df["Profit"].mean():,.1f}', "Avg Profit / Order", "linear-gradient(90deg,#fbbf24,#f472b6)")
    kpi(k4, "✅", f'{(df["Profit"] > 0).mean() * 100:.1f}%', "Profitable Orders", "linear-gradient(90deg,#60a5fa,#a78bfa)")

    st.markdown("---")
    monthly = df.groupby(["OrderYear", "OrderMonth"], as_index=False)["Profit"].sum()
    monthly["Period"] = monthly["OrderYear"].astype(str) + "-" + monthly["OrderMonth"].astype(str).str.zfill(2)
    st.plotly_chart(style_fig(px.line(monthly, x="Period", y="Profit", title="Profit Trend Over Time")), use_container_width=True)

with tab4:
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(style_fig(px.bar(df.groupby("Category", as_index=False)["Profit"].sum(), x="Category", y="Profit", color="Category", title="Profit by Category")), use_container_width=True)
    with e2:
        st.plotly_chart(style_fig(px.bar(df.groupby("Region", as_index=False)["Profit"].sum(), x="Region", y="Profit", color="Region", title="Profit by Region")), use_container_width=True)

    st.plotly_chart(style_fig(px.scatter(df.sample(min(2000, len(df))), x="Sales", y="Profit", color="Category", opacity=0.6, title="Sales vs Profit")), use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    cats = df["Category"].unique()
    means = [df.loc[df["Category"] == c, "Discount"].mean() for c in cats]
    ax.bar(cats, means, color=PALETTE[: len(cats)])
    ax.set_title("Average Discount by Category", color="white")
    style_mpl(fig, ax)
    st.pyplot(fig)
    mpl_b64 = fig_to_base64(fig)

with tab5:
    st.header("Customer & Delivery Analysis")
    e3, e4 = st.columns(2)
    with e3:
        if 'Segment' in df.columns:
            st.plotly_chart(style_fig(px.pie(df, names="Segment", values="Sales", title="Sales by Customer Segment", hole=0.4)), use_container_width=True)
    with e4:
        if 'Ship Mode' in df.columns:
            st.plotly_chart(style_fig(px.histogram(df, x="Ship Mode", color="Category", title="Ship Mode Usage")), use_container_width=True)

with tab6:
    play_voice("training")
    st.header("Train & Compare Models")
    if st.button("🚀 Train 3 Models", type="primary"):
        X_train, X_test, y_train, y_test = sales.split(df)
        prog = st.progress(0, text="Training Linear Regression…")
        with st.status("Training models…", expanded=True) as s:
            pipes, times = {}, {}
            models = sales.get_models()
            for i, (name, model) in enumerate(models.items()):
                st.write(f"Step {i + 1}/{len(models)} — {name}")
                pipe = Pipeline([("prep", sales._preprocessor()), ("model", model)])
                t0 = time.perf_counter()
                pipe.fit(X_train, y_train)
                times[name] = round(time.perf_counter() - t0, 3)
                pipes[name] = pipe
                prog.progress((i + 1) / len(models), text=f"Trained {name}")
            s.update(label="All models trained ✅", state="complete")
        prog.empty()

        results = sales.evaluate(pipes, X_test, y_test)

        # SAVE MODEL (Assignment Requirement)
        best_name = results.iloc[0]["Model"]
        sales.save_model(pipes[best_name], "sales_best_model.pkl")

        st.session_state["sales_pipes"] = pipes
        st.session_state["sales_results"] = results
        st.session_state["sales_splits"] = (X_train, X_test, y_train, y_test)
        st.session_state["sales_times"] = times
        st.session_state["sales_autoloaded"] = False
        st.toast("Training complete & model saved ✅", icon="🤖")
        play_voice("complete")
        st.rerun()

    if "sales_results" in st.session_state:
        results = st.session_state["sales_results"]
        pipes = st.session_state["sales_pipes"]
        X_train, X_test, y_train, y_test = st.session_state["sales_splits"]

        if st.session_state.get("sales_autoloaded"):
            st.info("Pehle se saved model load hua hai. Sab 3 models compare karne ke liye **Train 3 Models** dabayein.")

        st.dataframe(results, use_container_width=True)
        best_name = results.iloc[0]["Model"]
        best_banner(best_name, f"R² = {results.iloc[0]['R²']}")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(style_fig(px.bar(results, x="Model", y="R²", color="Model", title="R² Comparison", text="R²")), use_container_width=True)
        with c2:
            st.plotly_chart(style_fig(px.bar(results, x="Model", y="RMSE", color="Model", title="RMSE Comparison (lower is better)", text="RMSE")), use_container_width=True)

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

        st.markdown("### Download Report")
        metrics = {r["Model"]: f"R²={r['R²']}, RMSE={r['RMSE']}, MAE={r['MAE']}" for _, r in results.iterrows()}
        metrics["Best model"] = best_name
        metrics["Rows analysed"] = f"{len(df):,}"
        figs = [b for b in [mpl_b64, avp_b64] if b]
        html = build_html_report("Sales Analytics & Profit Prediction", "Regression models trained on retail order data.", metrics, figs, "Predictions are decision support.")
        st.download_button("📥 Download Sales Report", html, file_name="sales_report.html", mime="text/html", use_container_width=True)
    else:
        st.info("Press **Train 3 Models** to compare regressors.")

with tab_cmp:
    st.header("🆚 Model Comparison")
    st.caption("Teeno regression models ka R², error, predictions, bias aur training time ek jagah.")
    if "sales_results" not in st.session_state:
        st.warning("Train models first (🤖 Model Evaluation tab).")
    else:
        cmp_res = st.session_state["sales_results"].copy()
        cmp_pipes = st.session_state["sales_pipes"]
        _, cmp_Xt, _, cmp_yt = st.session_state["sales_splits"]
        higher_better = ["R²"]
        lower_better = ["RMSE", "MAE"]
        metric_cols = higher_better + lower_better
        for m in metric_cols:
            cmp_res[m] = pd.to_numeric(cmp_res[m], errors="coerce")
        color_map = {n: MODEL_COLORS[i % len(MODEL_COLORS)] for i, n in enumerate(cmp_res["Model"])}

        if len(cmp_res) < 2:
            st.info("Abhi sirf ek saved model load hai. Teeno ka muqabla karne ke liye **Model Evaluation** tab mein **Train 3 Models** dabayein.")

        if cmp_res[metric_cols].isna().all().any():
            st.error(
                "Kisi metric (R², RMSE ya MAE) ki saari values NaN hain. "
                "Data mein missing values check karein, ya models/sales_best_model.pkl delete "
                "kar ke **Train 3 Models** dobara dabayein."
            )

        # ---- Winners ----
        st.subheader("🏆 Har metric ka winner")
        wcols = st.columns(len(metric_cols))
        for col, m in zip(wcols, metric_cols):
            if cmp_res[m].notna().any():
                idx = cmp_res[m].idxmax() if m in higher_better else cmp_res[m].idxmin()
                top = cmp_res.loc[idx]
                hint = "sab se zyada" if m in higher_better else "sab se kam"
                col.metric(f"{m} ({hint})", top["Model"], f"{top[m]:,.3f}", delta_color="off")
            else:
                col.metric(m, "—")

        ranks = pd.concat([
            cmp_res.set_index("Model")[higher_better].rank(ascending=False, method="min"),
            cmp_res.set_index("Model")[lower_better].rank(ascending=True, method="min"),
        ], axis=1)
        rank_s = ranks.mean(axis=1).sort_values()
        if rank_s.notna().any():
            st.success(f"🥇 Overall best: **{rank_s.index[0]}** (average rank {rank_s.iloc[0]:.2f}, 1 = sab se behtar)")

        # ---- Table ----
        st.subheader("📋 Metrics table (best value highlighted)")
        hl = "background-color:#34d399;color:#111;font-weight:bold"
        styled = (cmp_res.style
                  .highlight_max(subset=higher_better, props=hl)
                  .highlight_min(subset=lower_better, props=hl)
                  .format({m: "{:,.4f}" for m in metric_cols}, na_rep="—"))
        st.dataframe(styled, use_container_width=True)
        st.download_button("📥 Download comparison (CSV)", cmp_res.to_csv(index=False),
                           file_name="sales_model_comparison.csv", mime="text/csv")

        # ---- R² bar + Error bars ----
        g1, g2 = st.columns(2)
        with g1:
            fig_r2 = px.bar(cmp_res, x="Model", y="R²", color="Model", text="R²",
                            title="R² (zyada behtar)", color_discrete_map=color_map)
            st.plotly_chart(style_fig(fig_r2), use_container_width=True)
        with g2:
            err_melt = cmp_res.melt(id_vars="Model", value_vars=lower_better, var_name="Error", value_name="Value")
            fig_err = px.bar(err_melt, x="Model", y="Value", color="Error", barmode="group",
                             title="RMSE & MAE (kam behtar)")
            st.plotly_chart(style_fig(fig_err), use_container_width=True)

        # ---- Predictions ----
        preds = {name: np.asarray(pipe.predict(cmp_Xt)) for name, pipe in cmp_pipes.items()}
        y_arr = np.asarray(cmp_yt)

        st.subheader("🎯 Actual vs Predicted (teeno models)")
        rng = np.random.default_rng(42)
        pick = rng.choice(len(y_arr), size=min(1500, len(y_arr)), replace=False)
        fig_avp = go.Figure()
        for name, p in preds.items():
            fig_avp.add_trace(go.Scatter(x=y_arr[pick], y=p[pick], mode="markers", name=name,
                                         opacity=0.45, marker=dict(size=5, color=color_map.get(name))))
        lo = float(min(y_arr.min(), *[p.min() for p in preds.values()]))
        hi = float(max(y_arr.max(), *[p.max() for p in preds.values()]))
        fig_avp.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                     line=dict(dash="dash", color="#9ca3af")))
        fig_avp.update_layout(xaxis_title="Actual Profit", yaxis_title="Predicted Profit")
        st.plotly_chart(style_fig(fig_avp), use_container_width=True)

        # ---- Residuals ----
        st.subheader("📏 Residuals (Actual − Predicted)")
        fig_res = go.Figure()
        for name, p in preds.items():
            fig_res.add_trace(go.Box(y=(y_arr - p)[pick], name=name, marker_color=color_map.get(name), boxmean=True))
        fig_res.update_layout(yaxis_title="Residual ($)", showlegend=False)
        st.plotly_chart(style_fig(fig_res), use_container_width=True)
        st.caption("Box jitna chhota aur 0 ke qareeb ho, model utna behtar. Lambi whiskers matlab kuch orders mein bohat bara error.")

        # ---- Bias table ----
        bias_rows = []
        for name, p in preds.items():
            resid = y_arr - p
            bias_rows.append({
                "Model": name,
                "Average error / bias ($)": round(float(resid.mean()), 2),
                "Sab se bara under-prediction ($)": round(float(resid.max()), 2),
                "Sab se bara over-prediction ($)": round(float(-resid.min()), 2),
            })
        st.dataframe(pd.DataFrame(bias_rows), use_container_width=True)
        st.caption("Bias positive ho to model aam tor par profit **kam** batata hai, negative ho to **zyada** batata hai.")

        # ---- Training time ----
        times = st.session_state.get("sales_times")
        if times:
            st.subheader("⏱️ Training time")
            t_df = pd.DataFrame({"Model": list(times.keys()), "Seconds": list(times.values())})
            st.plotly_chart(style_fig(px.bar(t_df, x="Model", y="Seconds", color="Model", text="Seconds",
                                             color_discrete_map=color_map, title="Training Time (seconds)")),
                            use_container_width=True)

        # ---- Verdict ----
        st.subheader("📝 Natija")
        if cmp_res[metric_cols].notna().all().all() or all(cmp_res[m].notna().any() for m in metric_cols):
            b_r2 = cmp_res.loc[cmp_res["R²"].idxmax()]
            b_rmse = cmp_res.loc[cmp_res["RMSE"].idxmin()]
            b_mae = cmp_res.loc[cmp_res["MAE"].idxmin()]
            st.markdown(
                f"- **Best fit (R²):** {b_r2['Model']} ({b_r2['R²']:.3f})\n"
                f"- **Sab se kam bara errors (RMSE):** {b_rmse['Model']} ({b_rmse['RMSE']:,.2f})\n"
                f"- **Sab se kam average error (MAE):** {b_mae['Model']} ({b_mae['MAE']:,.2f})"
            )
        else:
            st.warning("Natija abhi nahi dikha sakte, kyunke metrics NaN hain. Upar wali error dekhein.")

with tab7:
    if "sales_pipes" in st.session_state:
        best_pipe = st.session_state["sales_pipes"][st.session_state["sales_results"].iloc[0]["Model"]]
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
            result_card("Predicted Profit", f"${profit:,.2f}", tone="safe" if profit > 0 else "risk")
            st.toast(f"Predicted profit: ${profit:,.2f}", icon="💰")
    else:
        st.warning("Go to 'Model Evaluation' tab and train models first.")

render_footer()