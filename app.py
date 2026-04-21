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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; }

.top-banner {
    background: #1565C0;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.top-banner h1 { color: #FFFFFF !important; font-size: 2rem !important; font-weight: 700 !important; margin: 0 !important; }
.top-banner p  { color: #BBDEFB !important; font-size: 0.95rem !important; margin: 4px 0 0 0 !important; }

.section-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #1565C0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #1565C0;
}

.summary-table { width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden; box-shadow:0 1px 6px rgba(0,0,0,0.08); }
.summary-table thead tr { background:#1565C0; color:white; }
.summary-table thead th { padding:11px 16px; text-align:left; font-size:0.83rem; font-weight:600; letter-spacing:0.05em; }
.summary-table tbody tr { border-bottom:1px solid #E3F2FD; }
.summary-table tbody tr:nth-child(odd)  { background:#F5F9FF; }
.summary-table tbody tr:nth-child(even) { background:#FFFFFF; }
.summary-table tbody td { padding:10px 16px; font-size:0.87rem; color:#1A1A2E; }
.summary-table tbody td:first-child { font-weight:600; color:#1565C0; }

.result-churn {
    background:#FFF3F3; border:1.5px solid #EF5350; border-radius:12px;
    padding:28px 20px; text-align:center;
}
.result-churn .ri { font-size:3rem; margin-bottom:8px; }
.result-churn .rt { font-size:1.35rem; font-weight:700; color:#C62828; margin-bottom:4px; }
.result-churn .rp { font-size:2.2rem; font-weight:700; color:#EF5350; margin-bottom:4px; }
.result-churn .rl { font-size:0.8rem; color:#C62828; }
.result-churn .rn { font-size:0.8rem; color:#B71C1C; background:#FFEBEE; border-radius:6px; padding:8px 12px; margin-top:10px; }

.result-safe {
    background:#F1FFF4; border:1.5px solid #43A047; border-radius:12px;
    padding:28px 20px; text-align:center;
}
.result-safe .ri { font-size:3rem; margin-bottom:8px; }
.result-safe .rt { font-size:1.35rem; font-weight:700; color:#1B5E20; margin-bottom:4px; }
.result-safe .rp { font-size:2.2rem; font-weight:700; color:#43A047; margin-bottom:4px; }
.result-safe .rl { font-size:0.8rem; color:#1B5E20; }
.result-safe .rn { font-size:0.8rem; color:#1B5E20; background:#E8F5E9; border-radius:6px; padding:8px 12px; margin-top:10px; }

.metric-row { display:flex; gap:10px; margin-top:16px; }
.metric-chip { flex:1; text-align:center; padding:10px 6px; border-radius:8px; background:#EEF4FF; border:1px solid #BBDEFB; }
.metric-chip .cv { font-size:1.2rem; font-weight:700; color:#1565C0; }
.metric-chip .cl { font-size:0.7rem; color:#555; margin-top:2px; }

.placeholder-box {
    border:2px dashed #BBDEFB; border-radius:12px;
    padding:52px 20px; text-align:center; background:#F5F9FF;
}
.placeholder-box .pi  { font-size:3rem; }
.placeholder-box .pt  { font-size:1rem; font-weight:600; color:#1565C0; margin-top:12px; }
.placeholder-box .pd  { font-size:0.82rem; color:#666; margin-top:8px; }

.info-box {
    background:#EEF4FF; border-left:4px solid #1565C0;
    border-radius:0 8px 8px 0; padding:11px 14px;
    font-size:0.83rem; color:#1A1A2E; margin-top:14px;
}

[data-testid="stSidebar"] { background:#F0F6FF; border-right:1px solid #BBDEFB; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider  label {
    font-weight:600 !important; color:#1A1A2E !important; font-size:0.84rem !important;
}
.sidebar-header {
    background:#1565C0; color:white; border-radius:10px;
    padding:13px 16px; margin-bottom:20px; text-align:center;
    font-weight:700; font-size:0.95rem; letter-spacing:0.03em;
}
.sidebar-tip {
    background:#E3F2FD; border-radius:8px; padding:9px 12px;
    font-size:0.74rem; color:#1565C0; margin-top:14px;
    text-align:center; border:1px solid #BBDEFB;
}

div[data-testid="stButton"] > button {
    background:#1565C0 !important; color:white !important;
    border:none !important; border-radius:8px !important;
    padding:12px 0 !important; font-size:0.93rem !important;
    font-weight:600 !important; width:100%;
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
except Exception as e:
    model_loaded = False

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div style="font-size:2.6rem">✈️</div>
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

        # Probability chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 2.2))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F8FBFF")
        bars = ax.barh(["Retention", "Churn Risk"], [ret_pct, churn_pct],
                       color=["#43A047", "#EF5350"], height=0.45,
                       edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, [ret_pct, churn_pct]):
            ax.text(min(val + 1.5, 90), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=11,
                    fontweight="bold", color="#1A1A2E")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", fontsize=9, color="#555")
        ax.set_title("Churn vs Retention Probability", fontsize=11,
                     fontweight="bold", color="#1565C0", pad=8)
        ax.tick_params(labelsize=9, colors="#444")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#DDD")
        ax.spines["bottom"].set_color("#DDD")
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
    "<div style='text-align:center;font-size:0.74rem;color:#888;margin-top:6px;'>"
    "B.Tech Gen AI &nbsp;·&nbsp; 2nd Semester &nbsp;·&nbsp; Final Project &nbsp;·&nbsp; 2026"
    "</div>", unsafe_allow_html=True
)