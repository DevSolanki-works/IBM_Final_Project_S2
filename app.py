import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- DARK MODE COOL UI CSS ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #E2E8F0;
}

/* Override Streamlit's default background */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0F172A, #020617);
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ===== TOP BANNER - GLASSMORPHIC DARK ===== */
.top-banner {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 32px;
    padding: 28px 36px;
    margin-bottom: 30px;
    display: flex;
    align-items: center;
    gap: 24px;
    box-shadow: 0 25px 40px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(6, 182, 212, 0.1) inset;
}
.top-banner h1 {
    color: #F0F9FF !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.01em;
    text-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
}
.top-banner p {
    color: #94A3B8 !important;
    font-size: 1rem !important;
    margin: 0 !important;
    font-weight: 400;
}
.top-banner-icon {
    font-size: 3.2rem;
    filter: drop-shadow(0 0 15px #06B6D4);
}

/* ===== SECTION TITLES ===== */
.section-title {
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid;
    border-image: linear-gradient(90deg, #06B6D4, #8B5CF6) 1;
    border-bottom-style: solid;
    border-bottom-color: #1E293B;
    color: #CBD5E1;
}

/* ===== SUMMARY TABLE - DARK NEON ===== */
.summary-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 20px;
    overflow: hidden;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid #1E293B;
    box-shadow: 0 15px 30px -10px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(6, 182, 212, 0.1) inset;
}
.summary-table thead tr {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border-bottom: 1px solid #06B6D4;
}
.summary-table thead th {
    padding: 14px 18px;
    text-align: left;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #06B6D4;
}
.summary-table tbody tr {
    transition: background 0.2s ease;
}
.summary-table tbody tr:hover {
    background: rgba(56, 189, 248, 0.08) !important;
}
.summary-table tbody tr:not(:last-child) {
    border-bottom: 1px solid #1E293B;
}
.summary-table tbody td {
    padding: 12px 18px;
    font-size: 0.95rem;
    color: #E2E8F0;
}
.summary-table tbody td:first-child {
    font-weight: 600;
    color: #38BDF8;
    background-color: rgba(15, 23, 42, 0.4);
}
.summary-table tbody tr:nth-child(odd) {
    background: rgba(15, 23, 42, 0.3);
}
.summary-table tbody tr:nth-child(even) {
    background: rgba(30, 41, 59, 0.3);
}

/* ===== RESULT CARDS - DARK WITH GLOW ===== */
.result-card {
    border-radius: 28px;
    padding: 30px 24px;
    text-align: center;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 25px 40px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(6, 182, 212, 0.15) inset;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.result-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 30px 50px -12px #000000, 0 0 20px rgba(6, 182, 212, 0.2);
}

.result-churn {
    background: rgba(127, 29, 29, 0.2);
    border: 1px solid #EF4444;
    border-left: 6px solid #EF4444;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.1);
}
.result-safe {
    background: rgba(6, 95, 70, 0.2);
    border: 1px solid #10B981;
    border-left: 6px solid #10B981;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.1);
}

.result-icon {
    font-size: 3.2rem;
    margin-bottom: 12px;
    filter: drop-shadow(0 0 12px currentColor);
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}
.result-churn .result-title { color: #FCA5A5; text-shadow: 0 0 8px #EF4444; }
.result-safe .result-title { color: #6EE7B7; text-shadow: 0 0 8px #10B981; }

.result-prob {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 4px;
}
.result-churn .result-prob { color: #F87171; text-shadow: 0 0 15px #DC2626; }
.result-safe .result-prob { color: #34D399; text-shadow: 0 0 15px #059669; }

.result-label {
    font-size: 0.9rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    opacity: 0.9;
    margin-bottom: 16px;
}
.result-churn .result-label { color: #FCA5A5; }
.result-safe .result-label { color: #A7F3D0; }

.result-note {
    font-size: 0.85rem;
    padding: 12px 14px;
    border-radius: 40px;
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(4px);
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.05);
}
.result-churn .result-note { background: rgba(185, 28, 28, 0.2); color: #FECACA; border-color: #7F1D1D; }
.result-safe .result-note { background: rgba(6, 78, 59, 0.2); color: #D1FAE5; border-color: #064E3B; }

/* ===== METRIC CHIPS - NEON DARK ===== */
.metric-row {
    display: flex;
    gap: 14px;
    margin-top: 22px;
}
.metric-chip {
    flex: 1;
    text-align: center;
    padding: 14px 6px;
    border-radius: 20px;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid #1E293B;
    box-shadow: 0 8px 16px -6px #00000080, 0 0 0 1px rgba(56, 189, 248, 0.1) inset;
    transition: all 0.2s;
}
.metric-chip:hover {
    border-color: #06B6D4;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
}
.metric-chip .chip-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F0F9FF;
    text-shadow: 0 0 8px #38BDF8;
}
.metric-chip .chip-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-top: 4px;
}

/* ===== PLACEHOLDER BOX ===== */
.placeholder-box {
    border: 2px dashed #1E293B;
    border-radius: 32px;
    padding: 52px 20px;
    text-align: center;
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.placeholder-box .pi { font-size: 3.8rem; opacity: 0.5; filter: drop-shadow(0 0 10px #06B6D4); }
.placeholder-box .pt {
    font-size: 1.3rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-top: 16px;
}
.placeholder-box .pd {
    font-size: 0.9rem;
    color: #94A3B8;
    margin-top: 8px;
}

/* ===== INFO BOX ===== */
.info-box {
    background: rgba(2, 132, 199, 0.1);
    border-left: 4px solid #06B6D4;
    border-radius: 0 16px 16px 0;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #BAE6FD;
    margin-top: 18px;
    backdrop-filter: blur(8px);
    border: 1px solid #0E7490;
}

/* ===== SIDEBAR - DARK ELEGANCE ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(145deg, #0B1120 0%, #0F172A 100%);
    border-right: 1px solid #1E293B;
    box-shadow: 8px 0 20px -10px #000000;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-weight: 600 !important;
    color: #CBD5E1 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stSlider > div {
    background: transparent !important;
}
.sidebar-header {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    color: #F0F9FF;
    border-radius: 20px;
    padding: 18px 18px;
    margin-bottom: 24px;
    text-align: center;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    border: 1px solid #06B6D4;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.15);
    text-shadow: 0 0 8px #38BDF8;
}
.sidebar-tip {
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 14px 16px;
    font-size: 0.8rem;
    color: #BAE6FD;
    margin-top: 22px;
    text-align: center;
    border: 1px solid #1E293B;
    box-shadow: inset 0 0 10px rgba(6,182,212,0.05);
}

/* ===== BUTTON - NEON GLOW ===== */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%) !important;
    color: #0F172A !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 14px 0 !important;
    font-size: 1.05rem !important;
    width: 100%;
    box-shadow: 0 0 25px #06B6D4, 0 8px 18px rgba(0,0,0,0.5);
    transition: all 0.2s ease;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.2) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #22D3EE 0%, #60A5FA 100%) !important;
    transform: scale(1.02);
    box-shadow: 0 0 35px #38BDF8, 0 10px 25px #000000;
    color: #020617 !important;
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.98);
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    font-weight: 700;
    color: #38BDF8 !important;
    background: rgba(15,23,42,0.5);
    border-radius: 12px;
}
.streamlit-expanderContent {
    background: transparent;
    color: #CBD5E1;
}

/* ===== CHART ===== */
.stPlotlyChart, .stPyplot {
    border-radius: 20px !important;
    overflow: hidden;
    background: rgba(15, 23, 42, 0.3);
    backdrop-filter: blur(4px);
    border: 1px solid #1E293B;
}

/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #06B6D4; }
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
except Exception as e:
    model_loaded = False

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div class="top-banner-icon">✈️</div>
    <div>
        <h1>Customer Churn Predictor</h1>
        <p>Random Forest Classifier · Travel Industry · B.Tech Gen AI — Final Project</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ **model.pkl not found.** Run the Jupyter notebook first to generate the model file.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🔍 CUSTOMER DETAILS</div>', unsafe_allow_html=True)

    age             = st.slider("Age", 18, 80, 32, 1)
    frequent_flyer  = st.selectbox("Frequent Flyer", ["No", "Yes", "No Record"])
    annual_income   = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"], index=1)
    services_opted  = st.slider("Services Opted", 1, 6, 2, 1)
    account_synced  = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
    booked_hotel    = st.selectbox("Booked Hotel or Not", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀  Predict Churn", use_container_width=True)
    st.markdown('<div class="sidebar-tip">⚡ Fill all fields · Click Predict</div>', unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1], gap="large")

# LEFT – Summary table
with col_l:
    st.markdown('<div class="section-title">📋 CUSTOMER SUMMARY</div>', unsafe_allow_html=True)
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
        ℹ️ Adjust the customer details in the sidebar and click <strong style="color:#38BDF8;">Predict Churn</strong> to see the result.
    </div>
    """, unsafe_allow_html=True)

# RIGHT – Result
with col_r:
    st.markdown('<div class="section-title">📊 PREDICTION RESULT</div>', unsafe_allow_html=True)

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
            <div class="result-card result-churn">
                <div class="result-icon">⚠️</div>
                <div class="result-title">CHURN RISK DETECTED</div>
                <div class="result-prob">{churn_pct:.1f}%</div>
                <div class="result-label">Churn Probability</div>
                <div class="result-note">💡 This customer is at risk of leaving.<br>Consider a loyalty offer or personalised follow-up.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title">LOW CHURN RISK</div>
                <div class="result-prob">{ret_pct:.1f}%</div>
                <div class="result-label">Retention Probability</div>
                <div class="result-note">🎉 This customer is likely to stay.<br>Keep up the great service to maintain loyalty!</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip"><div class="chip-value">{churn_pct:.0f}%</div><div class="chip-label">Churn Risk</div></div>
            <div class="metric-chip"><div class="chip-value">{ret_pct:.0f}%</div><div class="chip-label">Retention</div></div>
            <div class="metric-chip"><div class="chip-value">{risk_lbl}</div><div class="chip-label">Risk Level</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Probability chart (dark themed)
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 2.2))
        fig.patch.set_facecolor("#0B1120")
        ax.set_facecolor("#0F172A")
        bars = ax.barh(["Retention", "Churn Risk"], [ret_pct, churn_pct],
                       color=["#10B981", "#EF4444"], height=0.5,
                       edgecolor="#1E293B", linewidth=2)
        for bar, val in zip(bars, [ret_pct, churn_pct]):
            ax.text(min(val + 1.8, 90), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=11,
                    fontweight="bold", color="#E2E8F0")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", fontsize=9, color="#94A3B8", fontweight=500)
        ax.set_title("Churn vs Retention Probability", fontsize=11,
                     fontweight="700", color="#F0F9FF", pad=8)
        ax.tick_params(labelsize=9, colors="#CBD5E1")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.grid(axis="x", color="#1E293B", linewidth=0.8, linestyle="--")
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.markdown("""
        <div class="placeholder-box">
            <div class="pi">📊</div>
            <div class="pt">No Prediction Yet</div>
            <div class="pd">Fill in the customer details in the sidebar<br>and click <strong style="color:#38BDF8;">Predict Churn</strong></div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
with st.expander("ℹ️  About this app", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🧠 Model**")
        st.write("Random Forest Classifier trained on Customertravel.csv — 954 customers, 6 features.")
    with c2:
        st.markdown("**📦 Features**")
        st.write("Age · FrequentFlyer · AnnualIncomeClass · ServicesOpted · AccountSynced · BookedHotel")
    with c3:
        st.markdown("**📈 Performance**")
        st.write("Accuracy ~85% · ROC-AUC ~0.88 · Balanced class weights applied")

st.markdown(
    "<div style='text-align:center;font-size:0.74rem;color:#475569;margin-top:8px;'>"
    "B.Tech Gen AI · 2nd Semester · Final Project · 2026"
    "</div>", unsafe_allow_html=True
)