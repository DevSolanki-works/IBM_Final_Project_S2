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

# ----------------------------- CUSTOM CSS --------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Main container padding */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ===== TOP BANNER ===== */
.top-banner {
    background: linear-gradient(135deg, #0B1120 0%, #1E3A8A 100%);
    border-radius: 24px;
    padding: 28px 36px;
    margin-bottom: 30px;
    display: flex;
    align-items: center;
    gap: 24px;
    box-shadow: 0 20px 35px -8px rgba(0, 27, 58, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.top-banner h1 {
    color: #FFFFFF !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.01em;
}
.top-banner p {
    color: #B0C7E9 !important;
    font-size: 1rem !important;
    margin: 0 !important;
    font-weight: 400;
    opacity: 0.9;
}
.top-banner-icon {
    font-size: 3.2rem;
    filter: drop-shadow(0 8px 12px rgba(0,0,0,0.2));
}

/* ===== SECTION TITLES ===== */
.section-title {
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid;
    border-image: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%) 1;
    border-bottom-style: solid;
    border-bottom-color: #E2E8F0;
    color: #1E293B;
}

/* ===== SUMMARY TABLE ===== */
.summary-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.08);
    background: white;
    border: 1px solid #F1F5F9;
}
.summary-table thead tr {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    color: white;
}
.summary-table thead th {
    padding: 14px 18px;
    text-align: left;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.summary-table tbody tr {
    transition: background 0.15s ease;
}
.summary-table tbody tr:hover {
    background: #F8FAFC !important;
}
.summary-table tbody tr:not(:last-child) {
    border-bottom: 1px solid #EDF2F7;
}
.summary-table tbody td {
    padding: 12px 18px;
    font-size: 0.95rem;
    color: #0F172A;
}
.summary-table tbody td:first-child {
    font-weight: 600;
    color: #2563EB;
    background-color: #F8FAFC;
}
.summary-table tbody tr:nth-child(even) td:first-child {
    background-color: #FFFFFF;
}
.summary-table tbody tr:nth-child(odd) {
    background: #FFFFFF;
}
.summary-table tbody tr:nth-child(even) {
    background: #F9FAFB;
}

/* ===== RESULT CARDS ===== */
.result-card {
    border-radius: 24px;
    padding: 30px 24px;
    text-align: center;
    backdrop-filter: blur(2px);
    box-shadow: 0 20px 35px -8px rgba(0, 0, 0, 0.12);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 28px 40px -12px rgba(0, 0, 0, 0.18);
}

.result-churn {
    background: linear-gradient(145deg, #FFF5F5 0%, #FEF2F2 100%);
    border: 1px solid #FECACA;
    border-left: 6px solid #DC2626;
}
.result-safe {
    background: linear-gradient(145deg, #F4FBF7 0%, #ECFDF5 100%);
    border: 1px solid #A7F3D0;
    border-left: 6px solid #059669;
}

.result-icon {
    font-size: 3.2rem;
    margin-bottom: 12px;
    filter: drop-shadow(0 6px 8px rgba(0,0,0,0.05));
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}
.result-churn .result-title { color: #B91C1C; }
.result-safe .result-title { color: #065F46; }

.result-prob {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 4px;
}
.result-churn .result-prob { color: #DC2626; }
.result-safe .result-prob { color: #059669; }

.result-label {
    font-size: 0.9rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.8;
    margin-bottom: 16px;
}
.result-churn .result-label { color: #991B1B; }
.result-safe .result-label { color: #065F46; }

.result-note {
    font-size: 0.85rem;
    padding: 12px 14px;
    border-radius: 40px;
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(4px);
    font-weight: 500;
}
.result-churn .result-note { background: #FEE2E2; color: #7F1D1D; }
.result-safe .result-note { background: #D1FAE5; color: #064E3B; }

/* ===== METRIC CHIPS ===== */
.metric-row {
    display: flex;
    gap: 14px;
    margin-top: 22px;
}
.metric-chip {
    flex: 1;
    text-align: center;
    padding: 14px 6px;
    border-radius: 18px;
    background: white;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.02);
    transition: all 0.15s;
}
.metric-chip:hover {
    border-color: #3B82F6;
    box-shadow: 0 6px 12px rgba(59,130,246,0.08);
}
.metric-chip .chip-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1E293B;
}
.metric-chip .chip-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748B;
    margin-top: 4px;
}

/* ===== PLACEHOLDER BOX ===== */
.placeholder-box {
    border: 2px dashed #CBD5E1;
    border-radius: 28px;
    padding: 52px 20px;
    text-align: center;
    background: #F8FAFC;
    backdrop-filter: blur(4px);
}
.placeholder-box .pi { font-size: 3.5rem; opacity: 0.7; }
.placeholder-box .pt {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1E293B;
    margin-top: 16px;
}
.placeholder-box .pd {
    font-size: 0.9rem;
    color: #64748B;
    margin-top: 8px;
}

/* ===== INFO BOX (under summary) ===== */
.info-box {
    background: #F0F9FF;
    border-left: 4px solid #0284C7;
    border-radius: 0 14px 14px 0;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #0C4A6E;
    margin-top: 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8FAFC 0%, #EFF6FF 100%);
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-weight: 600 !important;
    color: #0F172A !important;
    font-size: 0.85rem !important;
}
.sidebar-header {
    background: linear-gradient(135deg, #0B1120 0%, #1E3A8A 100%);
    color: white;
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 24px;
    text-align: center;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
    box-shadow: 0 6px 12px rgba(0,27,58,0.1);
}
.sidebar-tip {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 0.8rem;
    color: #1E293B;
    margin-top: 18px;
    text-align: center;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}

/* ===== BUTTON ===== */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 14px 0 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.2);
    transition: all 0.2s ease;
    letter-spacing: 0.02em;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    transform: scale(1.01);
    box-shadow: 0 10px 22px rgba(37, 99, 235, 0.3);
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.99);
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #1E293B;
}

/* ===== CHART STYLING (override) ===== */
.stPlotlyChart, .stPyplot {
    border-radius: 18px !important;
    overflow: hidden;
    box-shadow: 0 6px 14px rgba(0,0,0,0.04) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
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
    st.markdown('<div class="sidebar-header">🔍 Customer Details</div>', unsafe_allow_html=True)

    age             = st.slider("Age", 18, 80, 32, 1)
    frequent_flyer  = st.selectbox("Frequent Flyer", ["No", "Yes", "No Record"])
    annual_income   = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"], index=1)
    services_opted  = st.slider("Services Opted", 1, 6, 2, 1)
    account_synced  = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
    booked_hotel    = st.selectbox("Booked Hotel or Not", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀  Predict Churn", use_container_width=True)
    st.markdown('<div class="sidebar-tip">Fill in all fields above then<br>click <strong>Predict Churn</strong></div>', unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1], gap="large")

# LEFT – Summary table
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
        ℹ️ Adjust the customer details in the sidebar and click <strong>Predict Churn</strong> to see the result.
    </div>
    """, unsafe_allow_html=True)

# RIGHT – Result
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

        # Probability chart (styled consistently)
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 2.2))
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#F8FAFC")
        bars = ax.barh(["Retention", "Churn Risk"], [ret_pct, churn_pct],
                       color=["#059669", "#DC2626"], height=0.5,
                       edgecolor="white", linewidth=2)
        for bar, val in zip(bars, [ret_pct, churn_pct]):
            ax.text(min(val + 1.8, 90), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=11,
                    fontweight="bold", color="#0F172A")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", fontsize=9, color="#475569", fontweight=500)
        ax.set_title("Churn vs Retention Probability", fontsize=11,
                     fontweight="700", color="#0F172A", pad=8)
        ax.tick_params(labelsize=9, colors="#334155")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8, linestyle="--")
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
    "<div style='text-align:center;font-size:0.74rem;color:#94A3B8;margin-top:8px;'>"
    "B.Tech Gen AI · 2nd Semester · Final Project · 2026"
    "</div>", unsafe_allow_html=True
)