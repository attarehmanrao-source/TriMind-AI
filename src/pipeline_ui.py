# src/pipeline_ui.py
"""Reusable dynamic ML workflow UI (CSV -> columns -> encode -> scale -> train -> evaluate)."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import pipeline as pl
from src.ui import best_banner, kpi, section, style_fig
from src.utils import build_html_report

_GRADS = [
    "linear-gradient(90deg,#34d399,#60a5fa)",
    "linear-gradient(90deg,#a78bfa,#f0abfc)",
    "linear-gradient(90deg,#fbbf24,#f472b6)",
    "linear-gradient(90deg,#60a5fa,#a78bfa)",
]
_ICONS = ["🎯", "📏", "🔁", "⭐"]


def render_pipeline_workflow(uploaded_file, key_prefix: str = "mlp", start_num: int = 2) -> None:
    """Render the whole dynamic workflow for an uploaded CSV."""
    kp = key_prefix
    n = start_num
    file_key = f"{kp}-{uploaded_file.name}-{uploaded_file.size}"

    # ---------------- Load (only when the file changes) ----------------
    if st.session_state.get(f"{kp}_file_key") != file_key:
        try:
            st.session_state[f"{kp}_df"] = pl.load_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSV read nahi ho saki: {e}")
            return
        st.session_state[f"{kp}_file_key"] = file_key
        st.session_state.pop(f"{kp}_result", None)

    df_raw = st.session_state[f"{kp}_df"]
    if df_raw.empty:
        st.error("File mein koi rows nahi hain.")
        return

    # ---------------- Preview ----------------
    section(n, "Data Preview", "Columns ke naam aur ibtidai rows — dekhein ke data mein kya hai.")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "📄", f"{len(df_raw):,}", "Rows", _GRADS[0])
    kpi(k2, "🧱", f"{df_raw.shape[1]}", "Columns", _GRADS[1])
    kpi(k3, "🕳️", f"{int(df_raw.isna().sum().sum()):,}", "Missing Cells", _GRADS[2])
    kpi(k4, "♻️", f"{int(df_raw.duplicated().sum()):,}", "Duplicate Rows", _GRADS[3])

    n_show = st.slider("Kitni rows dekhni hain?", 5, 50, 10, key=f"{kp}_nshow_{file_key}")
    st.dataframe(df_raw.head(n_show), use_container_width=True)

    with st.expander("📋 Column names, types, missing aur unique values"):
        info = pd.DataFrame(
            {
                "Column": df_raw.columns,
                "Type": df_raw.dtypes.astype(str).values,
                "Missing": df_raw.isna().sum().values,
                "Unique": df_raw.nunique().values,
            }
        )
        st.dataframe(info, use_container_width=True, hide_index=True)

    # ---------------- Column management ----------------
    n += 1
    section(n, "Column Management", "Jin columns ki zaroorat nahi (jaise ID, Name) unhein delete karein.")
    drop_cols = st.multiselect(
        "Delete karne wale columns", list(df_raw.columns), key=f"{kp}_drop_{file_key}"
    )
    df = pl.drop_columns(df_raw, drop_cols)
    if df.shape[1] < 2:
        st.error("Kam az kam 2 columns rakhein (1 target + 1 feature).")
        return
    st.caption(f"Baqi columns: {df.shape[1]}  |  Deleted: {len(drop_cols)}")

    # ---------------- Target and data types ----------------
    n += 1
    section(n, "Target & Data Types", "Target (jo predict karna hai) chunein. Baqi columns text aur numeric mein baant diye jayenge.")
    cols_list = list(df.columns)
    target = st.selectbox(
        "🎯 Target column", cols_list, index=len(cols_list) - 1, key=f"{kp}_target_{file_key}"
    )
    if df[target].isna().any():
        st.warning(
            f"Target column mein {int(df[target].isna().sum())} missing values hain — "
            "woh rows training mein shamil nahi hongi."
        )

    features_df = df.drop(columns=[target])
    num_cols, txt_cols = pl.split_column_types(features_df)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**🔢 Numeric columns ({len(num_cols)})**")
        st.write(", ".join(num_cols) if num_cols else "—")
    with t2:
        st.markdown(f"**🔤 Text columns ({len(txt_cols)})**")
        st.write(", ".join(txt_cols) if txt_cols else "—")

    high_card = [c for c in txt_cols if df[c].nunique() > 50]
    if high_card:
        st.warning(
            "In text columns mein bohat zyada unique values hain (ID / naam / date ho sakte hain). "
            "Behtar hai inhein delete kar dein: " + ", ".join(high_card)
        )

    miss = pl.missing_summary(features_df)
    if not miss.empty:
        st.caption("Missing values automatically fill hongi (numeric: median, text: most frequent).")
        with st.expander("Missing values ki tafseel"):
            st.dataframe(miss, use_container_width=True)

    # ---------------- Encoding ----------------
    n += 1
    section(n, "Encoding", "Text data ko numbers mein badalne ka tareeqa chunein.")
    if txt_cols:
        enc_label = st.selectbox("Encoding method", list(pl.ENCODING_OPTIONS.keys()), key=f"{kp}_enc_{file_key}")
        st.caption(
            "One-Hot: har category ka alag column (linear models ke liye behtar). "
            "Label: har category ko ek number (tree models ke liye theek)."
        )
    else:
        enc_label = list(pl.ENCODING_OPTIONS.keys())[0]
        st.info("Features mein koi text column nahi — encoding ki zaroorat nahi.")
    encoding = pl.ENCODING_OPTIONS[enc_label]

    # ---------------- Scaling ----------------
    n += 1
    section(n, "Scaling", "Numeric data ko ek range mein laane ke liye.")
    if num_cols:
        sc_label = st.selectbox(
            "Scaling method", list(pl.SCALING_OPTIONS.keys()), index=1, key=f"{kp}_sc_{file_key}"
        )
        st.caption(
            "Standardization: mean 0, std 1.  Normalization: values 0 se 1 ke darmiyan.  "
            "Tree models (Random Forest, Decision Tree) ko scaling ki zaroorat nahi hoti."
        )
    else:
        sc_label = "None"
        st.info("Features mein koi numeric column nahi — scaling ki zaroorat nahi.")
    scaling = pl.SCALING_OPTIONS[sc_label]

    # ---------------- Model training ----------------
    n += 1
    section(n, "Model Training", "Model chunein aur train karein.")
    detected = pl.detect_task(df[target].dropna()) if df[target].notna().any() else "classification"
    task_opts = ["classification", "regression"]
    task = st.radio(
        "Problem type (auto-detected, aap badal sakte hain)",
        task_opts,
        index=task_opts.index(detected),
        horizontal=True,
        key=f"{kp}_task_{file_key}_{target}",
    )
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        st.error("Target column text hai — regression ke liye numeric target chahiye. Classification chunein.")
        return

    mc1, mc2 = st.columns([2, 1])
    with mc1:
        model_name = st.selectbox(
            "Model", list(pl.MODELS[task].keys()), key=f"{kp}_model_{file_key}_{task}"
        )
    with mc2:
        test_pct = st.slider("Test data %", 10, 40, 20, step=5, key=f"{kp}_test_{file_key}")

    if st.button("🚀 Train Model", type="primary", use_container_width=True, key=f"{kp}_train_{file_key}"):
        try:
            with st.spinner(f"Training {model_name}…"):
                result = pl.train_and_evaluate(
                    df,
                    target=target,
                    model_name=model_name,
                    task=task,
                    encoding=encoding,
                    scaling=scaling,
                    test_size=test_pct / 100,
                )
            result["enc_label"] = enc_label
            result["sc_label"] = sc_label
            st.session_state[f"{kp}_result"] = result
            st.toast("Training complete ✅", icon="🤖")
        except Exception as e:
            st.session_state.pop(f"{kp}_result", None)
            st.error(f"Training fail hui: {e}")

    # ---------------- Evaluation ----------------
    result = st.session_state.get(f"{kp}_result")
    if result is None:
        st.info("Settings chunein aur **Train Model** dabayein.")
        return

    n += 1
    section(n, "Evaluation & Predictions", "Metrics, charts aur test data par predictions.")

    metrics = result["metrics"]
    first_key = next(iter(metrics))
    best_banner(
        f"{result['model_name']} ({result['task']})",
        f"{first_key} = {metrics[first_key]:.4f}  |  Train rows: {result['n_train']:,}  |  Test rows: {result['n_test']:,}",
    )

    mcols = st.columns(len(metrics))
    for i, (col, (name, val)) in enumerate(zip(mcols, metrics.items())):
        shown = f"{val * 100:.2f}%" if result["task"] == "classification" else f"{val:,.4f}"
        kpi(col, _ICONS[i % len(_ICONS)], shown, name, _GRADS[i % len(_GRADS)])

    y_true, y_pred = result["y_test"], result["preds"]
    if result["label_encoder"] is not None:
        le = result["label_encoder"]
        actual_vals = le.inverse_transform(y_true.astype(int))
        pred_vals = le.inverse_transform(y_pred.astype(int))
    else:
        actual_vals, pred_vals = y_true, y_pred

    pred_df = pd.DataFrame({"Actual": actual_vals, "Predicted": pred_vals})
    if result["task"] == "regression":
        pred_df["Error"] = pred_df["Actual"] - pred_df["Predicted"]
    else:
        pred_df["Correct"] = np.where(pred_df["Actual"] == pred_df["Predicted"], "✅", "❌")

    g1, g2 = st.columns(2)
    with g1:
        if result["task"] == "classification":
            classes = result["classes"]
            fig = px.imshow(
                result["confusion_matrix"],
                x=classes,
                y=classes,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                color_continuous_scale="Purples",
                title="Confusion Matrix",
            )
        else:
            sample = pred_df.sample(min(2000, len(pred_df)), random_state=42)
            fig = px.scatter(sample, x="Actual", y="Predicted", opacity=0.55, title="Actual vs Predicted")
            lo = float(min(sample["Actual"].min(), sample["Predicted"].min()))
            hi = float(max(sample["Actual"].max(), sample["Predicted"].max()))
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="#f472b6"))
        st.plotly_chart(style_fig(fig), use_container_width=True, key=f"{kp}_chart1")

    with g2:
        imp = result["importance"]
        if imp is not None and not imp.empty:
            top = imp.head(15).sort_values("Importance")
            fig_imp = px.bar(top, x="Importance", y="Feature", orientation="h", title="Top Features")
            st.plotly_chart(style_fig(fig_imp), use_container_width=True, key=f"{kp}_chart2")
        else:
            st.info("Is model ke liye feature importance dastiyab nahi.")

    st.markdown("### 🔮 Predictions (test data)")
    st.dataframe(pred_df.head(200), use_container_width=True)
    st.caption(f"Pehli 200 rows dikhayi gayi hain, total {len(pred_df):,}.")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📥 Download Predictions (CSV)",
            pred_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{kp}_dl_csv",
        )
    with d2:
        report_metrics = {k: f"{v:.4f}" for k, v in metrics.items()}
        report_metrics["Model"] = result["model_name"]
        report_metrics["Target"] = result["target"]
        report_metrics["Encoding"] = result.get("enc_label", enc_label)
        report_metrics["Scaling"] = result.get("sc_label", sc_label)
        html = build_html_report(
            "ML Pipeline Builder",
            f"{result['task'].title()} model trained on an uploaded CSV file.",
            report_metrics,
            [],
            "Predictions are decision support.",
        )
        st.download_button(
            "📄 Download Report (HTML)",
            html,
            file_name="ml_pipeline_report.html",
            mime="text/html",
            use_container_width=True,
            key=f"{kp}_dl_html",
        )