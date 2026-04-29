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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; }

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

.section-title {
    font-size: 0.82rem; font-weight: 700; color: #1565C0;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 12px; padding-bottom: 6px;
    border-bottom: 2px solid #1565C0;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #EEF4FF;
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #555;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1565C0 !important;
    color: white !important;
}

/* Graph card */
.graph-card {
    background: #FFFFFF;
    border: 1px solid #DDEEFF;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.07);
}
.graph-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1565C0;
    margin-bottom: 4px;
}
.graph-desc {
    font-size: 0.8rem;
    color: #607080;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #E3F2FD;
}

.summary-table { width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden; box-shadow:0 1px 6px rgba(0,0,0,0.08); }
.summary-table thead tr { background:#1565C0; color:white; }
.summary-table thead th { padding:11px 16px; text-align:left; font-size:0.83rem; font-weight:600; }
.summary-table tbody tr { border-bottom:1px solid #E3F2FD; }
.summary-table tbody tr:nth-child(odd)  { background:#F5F9FF; }
.summary-table tbody tr:nth-child(even) { background:#FFFFFF; }
.summary-table tbody td { padding:10px 16px; font-size:0.87rem; color:#1A1A2E; }
.summary-table tbody td:first-child { font-weight:600; color:#1565C0; }

.result-churn { background:#FFF3F3; border:1.5px solid #EF5350; border-radius:12px; padding:26px 20px; text-align:center; }
.result-churn .ri { font-size:2.8rem; margin-bottom:6px; }
.result-churn .rt { font-size:1.3rem; font-weight:700; color:#C62828; margin-bottom:4px; }
.result-churn .rp { font-size:2.1rem; font-weight:700; color:#EF5350; margin-bottom:4px; }
.result-churn .rl { font-size:0.78rem; color:#C62828; }
.result-churn .rn { font-size:0.78rem; color:#B71C1C; background:#FFEBEE; border-radius:6px; padding:8px 12px; margin-top:10px; }

.result-safe { background:#F1FFF4; border:1.5px solid #43A047; border-radius:12px; padding:26px 20px; text-align:center; }
.result-safe .ri { font-size:2.8rem; margin-bottom:6px; }
.result-safe .rt { font-size:1.3rem; font-weight:700; color:#1B5E20; margin-bottom:4px; }
.result-safe .rp { font-size:2.1rem; font-weight:700; color:#43A047; margin-bottom:4px; }
.result-safe .rl { font-size:0.78rem; color:#1B5E20; }
.result-safe .rn { font-size:0.78rem; color:#1B5E20; background:#E8F5E9; border-radius:6px; padding:8px 12px; margin-top:10px; }

.metric-row { display:flex; gap:10px; margin-top:14px; }
.metric-chip { flex:1; text-align:center; padding:10px 4px; border-radius:8px; background:#EEF4FF; border:1px solid #BBDEFB; }
.metric-chip .cv { font-size:1.15rem; font-weight:700; color:#1565C0; }
.metric-chip .cl { font-size:0.68rem; color:#555; margin-top:2px; }

.placeholder-box { border:2px dashed #BBDEFB; border-radius:12px; padding:48px 20px; text-align:center; background:#F5F9FF; }
.placeholder-box .pi { font-size:2.8rem; }
.placeholder-box .pt { font-size:1rem; font-weight:600; color:#1565C0; margin-top:10px; }
.placeholder-box .pd { font-size:0.8rem; color:#666; margin-top:6px; }

.info-box { background:#EEF4FF; border-left:4px solid #1565C0; border-radius:0 8px 8px 0; padding:10px 14px; font-size:0.82rem; color:#1A1A2E; margin-top:12px; }

[data-testid="stSidebar"] { background:#F0F6FF; border-right:1px solid #BBDEFB; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { font-weight:600 !important; color:#1A1A2E !important; font-size:0.84rem !important; }
.sidebar-header { background:#1565C0; color:white; border-radius:10px; padding:13px 16px; margin-bottom:18px; text-align:center; font-weight:700; font-size:0.93rem; }
.sidebar-tip { background:#E3F2FD; border-radius:8px; padding:9px 12px; font-size:0.74rem; color:#1565C0; margin-top:12px; text-align:center; border:1px solid #BBDEFB; }

div[data-testid="stButton"] > button { background:#1565C0 !important; color:white !important; border:none !important; border-radius:8px !important; padding:11px 0 !important; font-size:0.92rem !important; font-weight:600 !important; width:100%; }
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
    st.markdown('<div class="sidebar-header">🔍 Customer Details</div>', unsafe_allow_html=True)
    age            = st.slider("Age", 18, 80, 32, 1)
    frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes", "No Record"])
    annual_income  = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"], index=1)
    services_opted = st.slider("Services Opted", 1, 6, 2, 1)
    account_synced = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
    booked_hotel   = st.selectbox("Booked Hotel or Not", ["No", "Yes"])
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀  Predict Churn", use_container_width=True)
    st.markdown('<div class="sidebar-tip">Fill in all fields above then<br>click <strong>Predict Churn</strong></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🏠  Predict", "📊  Visualizations", "ℹ️  About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-title">📋 Customer Summary</div>', unsafe_allow_html=True)
        rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in [
            ("Age", age), ("Frequent Flyer", frequent_flyer),
            ("Annual Income Class", annual_income), ("Services Opted", services_opted),
            ("Account Synced", account_synced), ("Booked Hotel", booked_hotel),
        ])
        st.markdown(f"""
        <table class="summary-table">
            <thead><tr><th>Field</th><th>Value</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div class="info-box">
            ℹ️ Adjust the customer details in the sidebar and click <strong>Predict Churn</strong>.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

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
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="ri">✅</div>
                    <div class="rt">LOW CHURN RISK</div>
                    <div class="rp">{ret_pct:.1f}%</div>
                    <div class="rl">Retention Probability</div>
                    <div class="rn">🎉 This customer is likely to stay.<br>Keep up the great service to maintain loyalty!</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-chip"><div class="cv">{churn_pct:.0f}%</div><div class="cl">Churn Risk</div></div>
                <div class="metric-chip"><div class="cv">{ret_pct:.0f}%</div><div class="cl">Retention</div></div>
                <div class="metric-chip"><div class="cv">{risk_lbl}</div><div class="cl">Risk Level</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 2.2))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#F8FBFF")
            bars = ax.barh(["Retention", "Churn Risk"], [ret_pct, churn_pct],
                           color=["#43A047", "#EF5350"], height=0.45, edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, [ret_pct, churn_pct]):
                ax.text(min(val + 1.5, 90), bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=11, fontweight="bold", color="#1A1A2E")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontsize=9, color="#555")
            ax.set_title("Churn vs Retention Probability", fontsize=11, fontweight="bold", color="#1565C0", pad=8)
            ax.tick_params(labelsize=9, colors="#444")
            for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#DDD"); ax.spines["bottom"].set_color("#DDD")
            ax.grid(axis="x", color="#E0E0E0", linewidth=0.6, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.markdown("""
            <div class="placeholder-box">
                <div class="pi">📊</div>
                <div class="pt">No Prediction Yet</div>
                <div class="pd">Fill in the customer details in the sidebar<br>and click <strong>Predict Churn</strong></div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📊 Model & Data Visualizations</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#607080;font-size:0.85rem;margin-bottom:20px;'>"
        "All charts below are generated from the trained Random Forest model on the Customertravel dataset.</p>",
        unsafe_allow_html=True
    )

    # Define all graphs with metadata
    graphs = [
        {
            "file":  "churn_distribution.png",
            "title": "📊  Churn Distribution",
            "desc":  "Shows the count and percentage split of churned (Target=1) vs retained (Target=0) customers in the dataset. Helps understand class imbalance.",
            "row":   1,
        },
        {
            "file":  "feature_analysis.png",
            "title": "📈  Feature Analysis vs Churn",
            "desc":  "6-panel chart showing churn rate broken down by each feature: Age, FrequentFlyer, AnnualIncomeClass, ServicesOpted, AccountSynced, and BookedHotel.",
            "row":   1,
        },
        {
            "file":  "correlation_heatmap.png",
            "title": "🔥  Correlation Heatmap",
            "desc":  "Pearson correlation matrix of all encoded features. Darker colors indicate stronger correlation. Helps identify which features are most related to the Target.",
            "row":   2,
        },
        {
            "file":  "confusion_matrix.png",
            "title": "🎯  Confusion Matrix",
            "desc":  "2×2 matrix of True Positives, True Negatives, False Positives, and False Negatives on the test split. Goal: maximize TP & TN, minimize FN.",
            "row":   2,
        },
        {
            "file":  "roc_curve.png",
            "title": "📉  ROC Curve (Receiver Operating Characteristic)",
            "desc":  "Plots True Positive Rate vs False Positive Rate at all classification thresholds. AUC ≈ 0.88 — well above the random classifier baseline (AUC = 0.5).",
            "row":   3,
        },
        {
            "file":  "feature_importance.png",
            "title": "⭐  Feature Importance Chart",
            "desc":  "Horizontal bar chart ranking all 6 input features by their importance score from the Random Forest. Higher score = more influence on churn prediction.",
            "row":   3,
        },
    ]

    # Render in 2-column rows
    for i in range(0, len(graphs), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            if i + j < len(graphs):
                g = graphs[i + j]
                with col:
                    st.markdown(f"""
                    <div class="graph-card">
                        <div class="graph-title">{g['title']}</div>
                        <div class="graph-desc">{g['desc']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(g["file"]):
                        img = Image.open(g["file"])
                        st.image(img, use_container_width=True)
                    else:
                        st.warning(
                            f"⚠️ `{g['file']}` not found. "
                            "Run the Jupyter notebook fully to generate all charts."
                        )

    # Full-width note
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#EEF4FF;border-left:4px solid #1565C0;border-radius:0 8px 8px 0;
                padding:12px 16px;font-size:0.82rem;color:#1A1A2E;">
        💡 <strong>Tip:</strong> All PNG files above are auto-generated when you run
        <code>customer_churn_prediction.ipynb</code> from top to bottom.
        Make sure all <code>.png</code> files are in the <strong>same folder</strong> as <code>app.py</code>.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">ℹ️ About This Application</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🧠 Model**")
        st.write("Random Forest Classifier trained on Customertravel.csv — 954 customers, 6 features, binary target.")
    with c2:
        st.markdown("**📦 Features Used**")
        st.write("Age · FrequentFlyer · AnnualIncomeClass · ServicesOpted · AccountSyncedToSocialMedia · BookedHotelOrNot")
    with c3:
        st.markdown("**📈 Performance**")
        st.write("Accuracy ~85% · ROC-AUC ~0.88 · Precision ~0.83 · Recall ~0.79 · class_weight='balanced'")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🗂️ Deployment Files**")
    st.code("app.py  |  model.pkl  |  requirements.txt  |  *.png charts", language="text")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;font-size:0.73rem;color:#888;margin-top:16px;'>"
    "B.Tech Gen AI &nbsp;·&nbsp; 2nd Semester &nbsp;·&nbsp; Final Project &nbsp;·&nbsp; 2026"
    "</div>", unsafe_allow_html=True
)