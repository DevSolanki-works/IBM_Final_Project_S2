import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — only functional bits, no forced colors
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-left: 2rem; padding-right: 2rem; }

.top-banner {
    background: #1565C0;
    border-radius: 12px;
    padding: 22px 30px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.top-banner h1 { color:#FFFFFF !important; font-size:1.9rem !important; font-weight:700 !important; margin:0 !important; }
.top-banner p  { color:#BBDEFB !important; font-size:0.9rem !important; margin:4px 0 0 0 !important; }

.graph-label {
    font-size: 1rem;
    font-weight: 700;
    color: #5B9BF5;
    margin-bottom: 4px;
    margin-top: 4px;
}
.graph-desc {
    font-size: 0.8rem;
    color: #8899AA;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2A3A4A;
}

.result-churn {
    background:#3D1515; border:1.5px solid #EF5350; border-radius:12px;
    padding:26px 20px; text-align:center;
}
.result-churn .ri { font-size:2.8rem; margin-bottom:6px; }
.result-churn .rt { font-size:1.3rem; font-weight:700; color:#FF6B6B; margin-bottom:4px; }
.result-churn .rp { font-size:2.1rem; font-weight:700; color:#EF5350; margin-bottom:4px; }
.result-churn .rl { font-size:0.78rem; color:#FF8A80; }
.result-churn .rn { font-size:0.78rem; color:#FFCDD2; background:#5C1A1A; border-radius:6px; padding:8px 12px; margin-top:10px; }

.result-safe {
    background:#0F3320; border:1.5px solid #43A047; border-radius:12px;
    padding:26px 20px; text-align:center;
}
.result-safe .ri { font-size:2.8rem; margin-bottom:6px; }
.result-safe .rt { font-size:1.3rem; font-weight:700; color:#69F0AE; margin-bottom:4px; }
.result-safe .rp { font-size:2.1rem; font-weight:700; color:#43A047; margin-bottom:4px; }
.result-safe .rl { font-size:0.78rem; color:#A5D6A7; }
.result-safe .rn { font-size:0.78rem; color:#C8E6C9; background:#1A4A2A; border-radius:6px; padding:8px 12px; margin-top:10px; }

.metric-row { display:flex; gap:10px; margin-top:14px; }
.metric-chip { flex:1; text-align:center; padding:10px 4px; border-radius:8px; background:#1A2A3A; border:1px solid #2A4A6A; }
.metric-chip .cv { font-size:1.15rem; font-weight:700; color:#5B9BF5; }
.metric-chip .cl { font-size:0.68rem; color:#7A8A9A; margin-top:2px; }

.placeholder-box { border:2px dashed #2A4A6A; border-radius:12px; padding:48px 20px; text-align:center; }
.placeholder-box .pi { font-size:2.8rem; }
.placeholder-box .pt { font-size:1rem; font-weight:600; color:#5B9BF5; margin-top:10px; }
.placeholder-box .pd { font-size:0.8rem; color:#7A8A9A; margin-top:6px; }

div[data-testid="stButton"] > button {
    background:#1565C0 !important; color:white !important;
    border:none !important; border-radius:8px !important;
    padding:11px 0 !important; font-size:0.92rem !important; font-weight:600 !important;
}
div[data-testid="stButton"] > button:hover { background:#0D47A1 !important; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoders"], data["features"]

try:
    model, encoders, features = load_model()
    model_loaded = True
except:
    model_loaded = False

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div style="font-size:2.5rem">✈️</div>
    <div>
        <h1>Customer Churn Prediction</h1>
        <p>Random Forest Classifier &nbsp;·&nbsp; Travel Industry &nbsp;·&nbsp; B.Tech Gen AI — Final Project</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ **model.pkl not found.** Run the Jupyter notebook first to generate the model file.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Customer Details")
    age            = st.slider("Age", 18, 80, 32, 1)
    frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes", "No Record"])
    annual_income  = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"], index=1)
    services_opted = st.slider("Services Opted", 1, 6, 2, 1)
    account_synced = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
    booked_hotel   = st.selectbox("Booked Hotel or Not", ["No", "Yes"])
    st.markdown("---")
    predict_btn = st.button("🚀  Predict Churn", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🏠  Predict", "📊  Visualizations"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.subheader("📋 Customer Summary")
        st.table(pd.DataFrame({
            "Field": ["Age", "Frequent Flyer", "Annual Income Class",
                      "Services Opted", "Account Synced", "Booked Hotel"],
            "Value": [str(age), frequent_flyer, annual_income,
                      str(services_opted), account_synced, booked_hotel]
        }))

    with col_r:
        st.subheader("📊 Prediction Result")

        if predict_btn:
            inp = pd.DataFrame([{
                "Age": age, "FrequentFlyer": frequent_flyer,
                "AnnualIncomeClass": annual_income, "ServicesOpted": services_opted,
                "AccountSyncedToSocialMedia": account_synced, "BookedHotelOrNot": booked_hotel,
            }])
            for col in ["FrequentFlyer", "AnnualIncomeClass", "AccountSyncedToSocialMedia", "BookedHotelOrNot"]:
                le  = encoders[col]
                val = inp[col].values[0]
                inp[col] = le.transform([val]) if val in le.classes_ else [0]
            inp = inp[features]

            pred      = model.predict(inp)[0]
            proba     = model.predict_proba(inp)[0]
            churn_pct = proba[1] * 100
            ret_pct   = proba[0] * 100
            risk_lbl  = "High" if churn_pct > 60 else ("Medium" if churn_pct > 35 else "Low")

            if pred == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <div class="ri">⚠️</div>
                    <div class="rt">CHURN RISK DETECTED</div>
                    <div class="rp">{churn_pct:.1f}%</div>
                    <div class="rl">Churn Probability</div>
                    <div class="rn">💡 This customer is at risk of leaving.<br>Consider a loyalty offer or personalised follow-up.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="ri">✅</div>
                    <div class="rt">LOW CHURN RISK</div>
                    <div class="rp">{ret_pct:.1f}%</div>
                    <div class="rl">Retention Probability</div>
                    <div class="rn">🎉 This customer is likely to stay.<br>Keep up the great service!</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-chip"><div class="cv">{churn_pct:.0f}%</div><div class="cl">Churn Risk</div></div>
                <div class="metric-chip"><div class="cv">{ret_pct:.0f}%</div><div class="cl">Retention</div></div>
                <div class="metric-chip"><div class="cv">{risk_lbl}</div><div class="cl">Risk Level</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 2.2))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            bars = ax.barh(["Retention", "Churn Risk"], [ret_pct, churn_pct],
                           color=["#43A047", "#EF5350"], height=0.45, edgecolor="#0E1117")
            for bar, val in zip(bars, [ret_pct, churn_pct]):
                ax.text(min(val + 1.5, 90), bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=11, fontweight="bold", color="white")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontsize=9, color="#8899AA")
            ax.set_title("Churn vs Retention Probability", fontsize=11, fontweight="bold", color="#5B9BF5", pad=8)
            ax.tick_params(labelsize=9, colors="#AAAAAA")
            for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#2A3A4A"); ax.spines["bottom"].set_color("#2A3A4A")
            ax.grid(axis="x", color="#1A2A3A", linewidth=0.8, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.markdown("""
            <div class="placeholder-box">
                <div class="pi">📊</div>
                <div class="pt">No Prediction Yet</div>
                <div class="pd">Fill in the customer details in the sidebar<br>and click <strong>Predict Churn</strong></div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Model & Data Visualizations")
    st.caption("All charts are generated from the trained Random Forest model on the Customertravel dataset.")
    st.markdown("---")

    graphs = [
        {
            "file":  "churn_distribution.png",
            "title": "📊  Graph 1 — Churn Distribution",
            "desc":  "Count and percentage split of churned (Target=1) vs retained (Target=0) customers. Shows class imbalance in the dataset.",
        },
        {
            "file":  "feature_analysis.png",
            "title": "📈  Graph 2 — Feature Analysis vs Churn",
            "desc":  "6-panel chart showing churn rate broken down by Age, FrequentFlyer, AnnualIncomeClass, ServicesOpted, AccountSynced, and BookedHotel.",
        },
        {
            "file":  "correlation_heatmap.png",
            "title": "🔥  Graph 3 — Correlation Heatmap",
            "desc":  "Pearson correlation matrix of all encoded features. Darker colors = stronger correlation with the Target variable.",
        },
        {
            "file":  "confusion_matrix.png",
            "title": "🎯  Graph 4 — Confusion Matrix",
            "desc":  "2×2 matrix of True Positives, True Negatives, False Positives, and False Negatives on the test set.",
        },
        {
            "file":  "roc_curve.png",
            "title": "📉  Graph 5 — ROC Curve",
            "desc":  "True Positive Rate vs False Positive Rate at all thresholds. AUC ≈ 0.88 — well above random classifier (AUC = 0.5).",
        },
        {
            "file":  "feature_importance.png",
            "title": "⭐  Graph 6 — Feature Importance",
            "desc":  "Ranks all 6 features by their importance score from the Random Forest. Higher = more influence on churn prediction.",
        },
    ]

    for i in range(0, len(graphs), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            if i + j < len(graphs):
                g = graphs[i + j]
                with col:
                    st.markdown(f'<div class="graph-label">{g["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="graph-desc">{g["desc"]}</div>', unsafe_allow_html=True)
                    if os.path.exists(g["file"]):
                        st.image(Image.open(g["file"]), use_container_width=True)
                    else:
                        st.warning(f"⚠️ `{g['file']}` not found — run the notebook to generate it.")
        st.markdown("<br>", unsafe_allow_html=True)

    st.info("💡 All PNG files are auto-generated when you run `customer_churn_prediction.ipynb` top to bottom. Make sure all `.png` files are in the same folder as `app.py`.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.73rem;color:#555;'>"
    "B.Tech Gen AI &nbsp;·&nbsp; 2nd Semester &nbsp;·&nbsp; Final Project &nbsp;·&nbsp; 2026"
    "</div>", unsafe_allow_html=True
)