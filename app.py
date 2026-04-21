import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
)

# ── Load Model ───────────────────────────────────────────────────────────────
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
    st.error(f"⚠️ Could not load model.pkl: {e}")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1565C0, #42A5F5);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 5px solid #1565C0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .result-churn {
        background: #ffebee;
        border: 2px solid #ef5350;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-no-churn {
        background: #e8f5e9;
        border: 2px solid #66bb6a;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
<div class="main-header">
    <h1>✈️ Customer Churn Prediction</h1>
    <p>Random Forest Classifier | Travel Industry</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar – Input ───────────────────────────────────────────────────────────
st.sidebar.title("🔍 Customer Details")
st.sidebar.markdown("Fill in the customer information below:")

age = st.sidebar.slider("Age", min_value=18, max_value=80, value=34, step=1)

frequent_flyer = st.sidebar.selectbox(
    "Frequent Flyer", options=["No", "Yes", "No Record"]
)

annual_income = st.sidebar.selectbox(
    "Annual Income Class", options=["Low Income", "Middle Income", "High Income"]
)

services_opted = st.sidebar.slider(
    "Services Opted", min_value=1, max_value=6, value=3, step=1
)

account_synced = st.sidebar.selectbox(
    "Account Synced to Social Media", options=["No", "Yes"]
)

booked_hotel = st.sidebar.selectbox(
    "Booked Hotel or Not", options=["No", "Yes"]
)

predict_btn = st.sidebar.button("🚀 Predict Churn", use_container_width=True)

# ── Main Content ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Customer Summary")
    st.markdown(f"""
    | Field | Value |
    |-------|-------|
    | **Age** | {age} |
    | **Frequent Flyer** | {frequent_flyer} |
    | **Annual Income Class** | {annual_income} |
    | **Services Opted** | {services_opted} |
    | **Account Synced** | {account_synced} |
    | **Booked Hotel** | {booked_hotel} |
    """)

with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn and model_loaded:
        # Build input dict and encode
        input_data = {
            "Age": age,
            "FrequentFlyer": frequent_flyer,
            "AnnualIncomeClass": annual_income,
            "ServicesOpted": services_opted,
            "AccountSyncedToSocialMedia": account_synced,
            "BookedHotelOrNot": booked_hotel,
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical columns using saved encoders
        cat_cols = ["FrequentFlyer", "AnnualIncomeClass", "AccountSyncedToSocialMedia", "BookedHotelOrNot"]
        for col in cat_cols:
            le = encoders[col]
            val = input_df[col].values[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                # Handle unseen label gracefully
                input_df[col] = 0

        # Reorder columns to match training
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        churn_prob = proba[1] * 100
        no_churn_prob = proba[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-churn">
                <h2>⚠️ CHURN RISK DETECTED</h2>
                <h3>Churn Probability: {churn_prob:.1f}%</h3>
                <p>This customer is likely to leave. Consider retention strategies.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-no-churn">
                <h2>✅ LOW CHURN RISK</h2>
                <h3>Retention Probability: {no_churn_prob:.1f}%</h3>
                <p>This customer is likely to stay. Keep up the good service!</p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bars = ax.barh(
            ["No Churn", "Churn"],
            [no_churn_prob, churn_prob],
            color=["#66bb6a", "#ef5350"],
            edgecolor="white",
            height=0.5,
        )
        for bar, val in zip(bars, [no_churn_prob, churn_prob]):
            ax.text(
                min(val + 1, 95), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontweight="bold", fontsize=12,
            )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Churn Probability Breakdown", fontweight="bold")
        ax.set_facecolor("#398bdc")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)

    elif predict_btn and not model_loaded:
        st.error("Model not available. Please upload model.pkl.")
    else:
        st.info("👈 Fill in the customer details and click **Predict Churn**")

# ── About Section ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    **Customer Churn Prediction App**  
    - **Algorithm**: Random Forest Classifier  
    - **Dataset**: Customertravel.csv (Travel industry customer data)  
    - **Target**: Predict whether a customer will churn (1) or not (0)

    **Features Used:**
    - `Age` – Customer age
    - `FrequentFlyer` – Whether the customer is a frequent flyer (Yes/No/No Record)
    - `AnnualIncomeClass` – Income bracket (Low/Middle/High)
    - `ServicesOpted` – Number of travel services opted (1–6)
    - `AccountSyncedToSocialMedia` – Whether account is synced to social media (Yes/No)
    - `BookedHotelOrNot` – Whether the customer booked a hotel (Yes/No)

    **Course**: Dev Solanki | Gen AI - B | Final Project
    """)

st.markdown(
    "<center><small>Built with ❤️ using Streamlit & Scikit-learn</small></center>",
    unsafe_allow_html=True,
)