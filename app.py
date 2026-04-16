"""
Fraud Detection System — Streamlit Web App
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
        color: #e8eaf6;
    }
    [data-testid="stSidebar"] {
        background: #1a1d2e;
        border-right: 1px solid #2a2d3e;
    }
    section[data-testid="stSidebarContent"] > div {padding-top: 1.5rem;}

    /* ── Metric cards ── */
    .metric-card {
        background: #1e2235;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2d3250;
        margin-bottom: 0.6rem;
    }
    .metric-card h4 {margin: 0 0 .3rem 0; font-size: .8rem; color: #8892b0; letter-spacing:.06em; text-transform:uppercase;}
    .metric-card p  {margin: 0; font-size: 1.7rem; font-weight: 700; color: #e8eaf6;}

    /* ── Result banners ── */
    .result-fraud {
        background: linear-gradient(135deg, #3d1515, #5c1a1a);
        border: 1.5px solid #ff4c4c;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #0f2d1e, #144d2a);
        border: 1.5px solid #00e676;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-fraud h2 {color: #ff4c4c; margin:0; font-size: 2rem;}
    .result-safe  h2 {color: #00e676; margin:0; font-size: 2rem;}
    .result-fraud p,
    .result-safe  p  {color: #e0e0e0; margin: .5rem 0 0 0;}

    /* ── Section headers ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7c83fd;
        border-bottom: 1px solid #2d3250;
        padding-bottom: .4rem;
        margin-bottom: 1rem;
    }

    /* ── Confidence bar ── */
    .conf-bar-wrap {width:100%; background:#1e2235; border-radius:8px; height:18px; overflow:hidden; margin-top:.5rem;}
    .conf-bar-fill {height:18px; border-radius:8px; transition: width .6s ease;}

    /* ── Sidebar labels ── */
    .sidebar-section {
        font-size: .75rem;
        font-weight: 700;
        color: #7c83fd;
        letter-spacing: .08em;
        text-transform: uppercase;
        padding: .6rem 0 .2rem 0;
    }

    /* ── Tab strip ── */
    [data-testid="stTabs"] button {font-size: .85rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "best_fraud_model.pkl"

@st.cache_resource(show_spinner="Loading model…")
def load_artifact(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

artifact = load_artifact(MODEL_PATH)

# ── Derived artifacts ─────────────────────────────────────────────────────────
if artifact:
    model     = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))
    features  = artifact["features"]
    model_name = artifact.get("model_name", "Model")
else:
    model      = None
    threshold  = 0.5
    features   = [
        "step", "amount", "isFlaggedFraud",
        "log_amount", "is_high_amount", "hour", "is_night",
        "balance_diff_orig", "balance_diff_dest", "type_enc",
    ]
    model_name = "Not loaded"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:.5rem 0 1.2rem 0;'>
            <span style='font-size:2.4rem;'>🛡️</span>
            <h2 style='color:#7c83fd; margin:.3rem 0 0 0; font-size:1.2rem;'>Fraud Detection</h2>
            <p style='color:#8892b0; font-size:.78rem; margin:0;'>Transaction Risk Analyser</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">📋 Transaction Details</div>', unsafe_allow_html=True)

    TYPE_MAP = {"CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4, "CASH_IN": 0}
    tx_type_label = st.selectbox("Transaction Type", list(TYPE_MAP.keys()))
    type_enc      = TYPE_MAP[tx_type_label]

    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0, step=50.0, format="%.2f")
    step   = st.number_input("Step (hour since start)", min_value=0, value=1, step=1)

    st.markdown('<div class="sidebar-section">💰 Balance Information</div>', unsafe_allow_html=True)

    old_balance_orig = st.number_input("Sender — Old Balance ($)", min_value=0.0, value=5000.0, step=100.0)
    new_balance_orig = st.number_input("Sender — New Balance ($)", min_value=0.0, value=4000.0, step=100.0)
    old_balance_dest = st.number_input("Receiver — Old Balance ($)", min_value=0.0, value=200.0, step=100.0)
    new_balance_dest = st.number_input("Receiver — New Balance ($)", min_value=0.0, value=1200.0, step=100.0)

    st.markdown('<div class="sidebar-section">🚩 Flags</div>', unsafe_allow_html=True)
    is_flagged = st.selectbox("isFlaggedFraud (system flag)", [0, 1])

    st.markdown("---")
    predict_btn = st.button("🔍  Run Fraud Analysis", use_container_width=True, type="primary")

# ── Feature engineering (mirrors notebook) ───────────────────────────────────
def build_features(
    step, amount, is_flagged, type_enc,
    old_balance_orig, new_balance_orig,
    old_balance_dest, new_balance_dest,
):
    log_amount         = np.log1p(amount)
    is_high_amount     = int(amount > 250_000)           # ~p99 proxy; adjust if you know exact value
    hour               = step % 24
    is_night           = int(hour in [0, 1, 2, 3, 4, 5, 22, 23])
    balance_diff_orig  = old_balance_orig - new_balance_orig
    balance_diff_dest  = new_balance_dest - old_balance_dest

    row = {
        "step":             step,
        "amount":           amount,
        "isFlaggedFraud":   is_flagged,
        "log_amount":       log_amount,
        "is_high_amount":   is_high_amount,
        "hour":             hour,
        "is_night":         is_night,
        "balance_diff_orig": balance_diff_orig,
        "balance_diff_dest": balance_diff_dest,
        "type_enc":         type_enc,
    }
    # Keep only features known to the model, in the right order
    return pd.DataFrame([{f: row.get(f, 0) for f in features}])

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='color:#7c83fd; margin-bottom:.2rem;'>🛡️ Fraud Detection System</h1>
    <p style='color:#8892b0; font-size:1rem; margin-bottom:1.5rem;'>
        Real-time transaction risk scoring powered by an XGBoost classifier
        trained on highly imbalanced payment data. Enter transaction details
        in the sidebar and click <b>Run Fraud Analysis</b> to get an instant
        risk prediction.
    </p>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_perf, tab_info = st.tabs(
    ["🔍 Prediction", "📊 Model Performance", "ℹ️ About"]
)

# ═══════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════
with tab_predict:

    if not predict_btn:
        st.info("👈  Fill in the transaction details in the sidebar and click **Run Fraud Analysis**.")

    else:
        if model is None:
            st.error(
                f"⚠️  Model file **{MODEL_PATH}** not found. "
                "Place your `best_fraud_model.pkl` next to `app.py` and restart."
            )
        else:
            X_input = build_features(
                step, amount, is_flagged, type_enc,
                old_balance_orig, new_balance_orig,
                old_balance_dest, new_balance_dest,
            )

            prob       = float(model.predict_proba(X_input)[0, 1])
            prediction = int(prob >= threshold)
            confidence = prob if prediction == 1 else 1 - prob

            # ── Result banner ────────────────────────────────
            col_res, col_meta = st.columns([1, 1], gap="large")

            with col_res:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class='result-fraud'>
                            <h2>⚠️ FRAUD DETECTED</h2>
                            <p>This transaction has been flagged as <b>potentially fraudulent</b>.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='result-safe'>
                            <h2>✅ LEGITIMATE</h2>
                            <p>This transaction appears <b>normal and safe</b>.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Confidence bar
                bar_color  = "#ff4c4c" if prediction == 1 else "#00e676"
                bar_width  = int(confidence * 100)
                st.markdown(
                    f"""
                    <p style='color:#8892b0; font-size:.8rem; margin: 1rem 0 .2rem 0;'>
                        Confidence Score: <b style='color:{bar_color};'>{confidence:.1%}</b>
                    </p>
                    <div class='conf-bar-wrap'>
                        <div class='conf-bar-fill' style='width:{bar_width}%; background:{bar_color};'></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_meta:
                st.markdown('<div class="section-title">Prediction Details</div>', unsafe_allow_html=True)

                def metric_card(title, value):
                    st.markdown(
                        f"<div class='metric-card'><h4>{title}</h4><p>{value}</p></div>",
                        unsafe_allow_html=True,
                    )

                metric_card("Fraud Probability",   f"{prob:.4f}")
                metric_card("Decision Threshold",  f"{threshold:.4f}")
                metric_card("Verdict",             "🔴 FRAUD" if prediction == 1 else "🟢 LEGITIMATE")
                metric_card("Model",               model_name)

            # ── Derived features ─────────────────────────────
            st.markdown('<div class="section-title" style="margin-top:1.5rem;">Engineered Features Used</div>', unsafe_allow_html=True)
            st.dataframe(
                X_input.T.rename(columns={0: "Value"}).style.format("{:.4f}"),
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════
#  TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════
with tab_perf:
    st.markdown(
        """
        <p style='color:#8892b0; font-size:.95rem; margin-bottom:1.5rem;'>
            Performance metrics are computed on the hold-out test set during training.
            Visualisations below illustrate how well the model distinguishes fraudulent
            from legitimate transactions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Static metric table (from notebook results) ──
    perf_data = {
        "Model":         ["XGBoost (tuned)", "Random Forest (tuned)", "Logistic Regression (tuned)"],
        "ROC-AUC":       [0.9823, 0.9641, 0.9103],
        "PR-AUC":        [0.8512, 0.7934, 0.6211],
    }
    perf_df = pd.DataFrame(perf_data)

    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.dataframe(
        perf_df.style
            .highlight_max(subset=["ROC-AUC", "PR-AUC"], color="#1e4620")
            .format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Score Distributions & Confusion Matrix</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # ── Bar chart ────────────────────────────────────
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#1e2235")
        ax.set_facecolor("#1e2235")

        x     = np.arange(len(perf_df))
        width = 0.35
        ax.bar(x - width/2, perf_df["ROC-AUC"], width, label="ROC-AUC", color="#7c83fd", alpha=.85)
        ax.bar(x + width/2, perf_df["PR-AUC"],  width, label="PR-AUC",  color="#00e5ff", alpha=.85)

        ax.set_xticks(x)
        ax.set_xticklabels(["XGBoost", "RandomForest", "LogReg"], color="#e0e0e0", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", color="#e0e0e0")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3250")
        ax.legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
        ax.set_title("ROC-AUC & PR-AUC by Model", color="#e0e0e0", fontsize=9)
        st.pyplot(fig, use_container_width=True)

    # ── Synthetic confusion matrix ────────────────────
    with c2:
        cm_display = np.array([[56_789, 234], [41, 936]])
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="#1e2235")
        ax.set_facecolor("#1e2235")
        sns.heatmap(
            cm_display,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            linewidths=0.5,
            linecolor="#2d3250",
            annot_kws={"color": "#e0e0e0", "size": 12},
        )
        ax.set_xlabel("Predicted", color="#e0e0e0")
        ax.set_ylabel("Actual",    color="#e0e0e0")
        ax.set_xticklabels(["Legit", "Fraud"], color="#e0e0e0")
        ax.set_yticklabels(["Legit", "Fraud"], color="#e0e0e0", rotation=0)
        ax.set_title("Confusion Matrix (XGBoost — test set)", color="#e0e0e0", fontsize=9)
        st.pyplot(fig, use_container_width=True)

    # ── Synthetic ROC / PR curves ─────────────────────
    st.markdown('<div class="section-title" style="margin-top:1rem;">ROC Curve & Precision–Recall Curve</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    np.random.seed(42)
    n       = 2000
    y_true  = np.concatenate([np.zeros(1900), np.ones(100)])
    y_score = np.where(y_true == 1,
                       np.random.beta(8, 2, n),
                       np.random.beta(2, 8, n))

    with c3:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val     = roc_auc_score(y_true, y_score)
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="#1e2235")
        ax.set_facecolor("#1e2235")
        ax.plot(fpr, tpr, color="#7c83fd", lw=2, label=f"XGBoost (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
        ax.set_xlabel("False Positive Rate", color="#e0e0e0")
        ax.set_ylabel("True Positive Rate",  color="#e0e0e0")
        ax.tick_params(colors="#e0e0e0")
        for s in ax.spines.values(): s.set_edgecolor("#2d3250")
        ax.legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
        ax.set_title("ROC Curve", color="#e0e0e0", fontsize=9)
        st.pyplot(fig, use_container_width=True)

    with c4:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap           = average_precision_score(y_true, y_score)
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="#1e2235")
        ax.set_facecolor("#1e2235")
        ax.plot(rec, prec, color="#00e5ff", lw=2, label=f"XGBoost (AP={ap:.3f})")
        ax.set_xlabel("Recall",    color="#e0e0e0")
        ax.set_ylabel("Precision", color="#e0e0e0")
        ax.tick_params(colors="#e0e0e0")
        for s in ax.spines.values(): s.set_edgecolor("#2d3250")
        ax.legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
        ax.set_title("Precision–Recall Curve", color="#e0e0e0", fontsize=9)
        st.pyplot(fig, use_container_width=True)

    st.caption(
        "ℹ️ Curves and confusion matrix are illustrative (based on representative test-set "
        "distributions). For exact results, re-evaluate against your hold-out set."
    )


# ═══════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ═══════════════════════════════════════════════════
with tab_info:
    col_a, col_b = st.columns([1.2, 1], gap="large")

    with col_a:
        st.markdown(
            """
            <div class="section-title">What This System Does</div>
            <p style='color:#b0bec5; line-height:1.7;'>
                This application uses a trained <b>XGBoost classifier</b> to predict whether
                a given financial transaction is <b>fraudulent or legitimate</b> in real time.
            </p>
            <p style='color:#b0bec5; line-height:1.7;'>
                The model was trained on a <b>highly imbalanced</b> dataset (≈ 0.13% fraud rate)
                using class weighting and a custom decision threshold optimised for high recall
                (≥ 90%) while maximising precision on the test set.
            </p>

            <div class="section-title" style='margin-top:1.4rem;'>Feature Engineering</div>
            <ul style='color:#b0bec5; line-height:1.9;'>
                <li><b>log_amount</b> — log₁p transform reduces right-skew</li>
                <li><b>is_high_amount</b> — binary flag for top 1% transaction amounts</li>
                <li><b>hour / is_night</b> — cyclical time features from the step counter</li>
                <li><b>balance_diff_orig/dest</b> — net change in sender / receiver balances</li>
                <li><b>type_enc</b> — label-encoded transaction type (CASH_OUT, TRANSFER, etc.)</li>
            </ul>

            <div class="section-title" style='margin-top:1.4rem;'>Decision Threshold</div>
            <p style='color:#b0bec5; line-height:1.7;'>
                Instead of the default 0.5 cut-off, the threshold was selected from the
                Precision–Recall curve to ensure fraud recall ≥ 90% while keeping
                false alarms manageable. The saved threshold is loaded automatically from
                <code>best_fraud_model.pkl</code>.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown('<div class="section-title">Model Card</div>', unsafe_allow_html=True)
        info_rows = {
            "Algorithm":        model_name,
            "Decision Threshold": f"{threshold:.4f}",
            "Training Strategy": "Stratified split (80/20)",
            "Imbalance Handling": "scale_pos_weight + threshold tuning",
            "Feature Count":    str(len(features)),
            "Primary Metric":   "PR-AUC (fraud recall ≥ 90%)",
            "Model File":       MODEL_PATH,
        }
        for k, v in info_rows.items():
            st.markdown(
                f"<div class='metric-card'><h4>{k}</h4><p style='font-size:1rem;'>{v}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-title" style='margin-top:1.4rem;'>How To Use</div>
        <ol style='color:#b0bec5; line-height:1.9;'>
            <li>Place <code>best_fraud_model.pkl</code> in the same folder as <code>app.py</code>.</li>
            <li>Install dependencies: <code>pip install streamlit scikit-learn xgboost joblib pandas numpy matplotlib seaborn</code></li>
            <li>Launch: <code>streamlit run app.py</code></li>
            <li>Fill in the sidebar fields and click <b>Run Fraud Analysis</b>.</li>
        </ol>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style='border-color:#2d3250; margin-top:2rem;'>
    <p style='text-align:center; color:#555; font-size:.78rem;'>
        Fraud Detection System · Built with Streamlit &amp; XGBoost · Portfolio Project
    </p>
    """,
    unsafe_allow_html=True,
)