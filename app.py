"""
Customer Churn Prediction & Retention Analytics - Streamlit Web Application

This interactive web application allows users to enter customer account attributes,
predict customer churn probability using the trained Logistic Regression pipeline,
categorize risk levels (Low, Medium, High), highlight observable risk drivers,
and generate rule-based retention actions.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value-high {
        font-size: 2.2rem;
        font-weight: 800;
        color: #dc2626;
    }
    .metric-value-med {
        font-size: 2.2rem;
        font-weight: 800;
        color: #d97706;
    }
    .metric-value-low {
        font-size: 2.2rem;
        font-weight: 800;
        color: #16a34a;
    }
    .driver-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .action-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .neutral-box {
        background-color: #f1f5f9;
        border-left: 4px solid #64748b;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_pipeline():
    """Loads the serialized Logistic Regression pipeline."""
    model_path = Path("models/logistic_regression_pipeline.joblib")
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path.resolve()}")
        st.stop()
    return joblib.load(model_path)


# Load pipeline
pipeline = load_model_pipeline()

# Header
st.markdown('<div class="main-title">📊 Customer Churn Prediction & Retention Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Risk Scoring & Rule-Based Customer Retention Engine</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("📋 Customer Profile Inputs")

# Quick Profile Presets
preset = st.sidebar.selectbox(
    "⚡ Quick Load Preset Profiles",
    ["Custom Input", "High-Risk Customer Profile", "Moderate-Risk Customer Profile", "Low-Risk Loyal Customer Profile"]
)

# Preset default values
if preset == "High-Risk Customer Profile":
    def_gender = "Female"
    def_senior = "No"
    def_partner = "No"
    def_dependents = "No"
    def_tenure = 2
    def_phone = "Yes"
    def_lines = "No"
    def_internet = "Fiber optic"
    def_sec = "No"
    def_backup = "No"
    def_dev = "No"
    def_tech = "No"
    def_tv = "No"
    def_movies = "No"
    def_contract = "Month-to-month"
    def_paperless = "Yes"
    def_payment = "Electronic check"
    def_monthly = 70.70
    def_total = 151.65
elif preset == "Moderate-Risk Customer Profile":
    def_gender = "Male"
    def_senior = "No"
    def_partner = "Yes"
    def_dependents = "No"
    def_tenure = 24
    def_phone = "Yes"
    def_lines = "Yes"
    def_internet = "DSL"
    def_sec = "No"
    def_backup = "Yes"
    def_dev = "Yes"
    def_tech = "No"
    def_tv = "Yes"
    def_movies = "No"
    def_contract = "One year"
    def_paperless = "Yes"
    def_payment = "Bank transfer (automatic)"
    def_monthly = 65.50
    def_total = 1572.00
elif preset == "Low-Risk Loyal Customer Profile":
    def_gender = "Male"
    def_senior = "No"
    def_partner = "Yes"
    def_dependents = "Yes"
    def_tenure = 60
    def_phone = "Yes"
    def_lines = "Yes"
    def_internet = "DSL"
    def_sec = "Yes"
    def_backup = "Yes"
    def_dev = "Yes"
    def_tech = "Yes"
    def_tv = "No"
    def_movies = "No"
    def_contract = "Two year"
    def_paperless = "No"
    def_payment = "Credit card (automatic)"
    def_monthly = 55.20
    def_total = 3312.00
else:
    def_gender = "Male"
    def_senior = "No"
    def_partner = "No"
    def_dependents = "No"
    def_tenure = 6
    def_phone = "Yes"
    def_lines = "No"
    def_internet = "Fiber optic"
    def_sec = "No"
    def_backup = "No"
    def_dev = "No"
    def_tech = "No"
    def_tv = "Yes"
    def_movies = "Yes"
    def_contract = "Month-to-month"
    def_paperless = "Yes"
    def_payment = "Electronic check"
    def_monthly = 89.90
    def_total = 539.40

with st.sidebar.expander("👤 Demographics", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"], index=0 if def_gender == "Male" else 1)
    senior = st.selectbox("Senior Citizen (Age 65+)", ["No", "Yes"], index=0 if def_senior == "No" else 1)
    partner = st.selectbox("Partner / Spouse", ["No", "Yes"], index=0 if def_partner == "No" else 1)
    dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if def_dependents == "No" else 1)

with st.sidebar.expander("📋 Contract & Account", expanded=True):
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_contract))
    tenure_months = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=def_tenure, step=1)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if def_paperless == "Yes" else 1)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment)
    )

with st.sidebar.expander("🌐 Services & Add-Ons", expanded=False):
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if def_phone == "Yes" else 1)
    if phone_service == "No":
        multiple_lines = "No phone service"
        st.info("Multiple Lines: No phone service")
    else:
        multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes"], index=0 if def_lines == "No" else 1)
        
    internet_service = st.selectbox("Internet Service Tier", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(def_internet))
    
    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        st.info("Internet Add-ons disabled (No Internet Service)")
    else:
        online_security = st.selectbox("Online Security Add-on", ["No", "Yes"], index=0 if def_sec == "No" else 1)
        online_backup = st.selectbox("Online Backup Add-on", ["No", "Yes"], index=0 if def_backup == "No" else 1)
        device_protection = st.selectbox("Device Protection Add-on", ["No", "Yes"], index=0 if def_dev == "No" else 1)
        tech_support = st.selectbox("Tech Support Add-on", ["No", "Yes"], index=0 if def_tech == "No" else 1)
        streaming_tv = st.selectbox("Streaming TV Service", ["No", "Yes"], index=0 if def_tv == "No" else 1)
        streaming_movies = st.selectbox("Streaming Movies Service", ["No", "Yes"], index=0 if def_movies == "No" else 1)

with st.sidebar.expander("💳 Billing & Charges", expanded=True):
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=125.0, value=float(def_monthly), step=0.5)
    
    # Auto-calculate default total charges helper
    est_total = round(float(monthly_charges * max(tenure_months, 1)), 2) if tenure_months > 0 else 0.0
    total_charges = st.number_input("Total Charges ($ to date)", min_value=0.0, max_value=10000.0, value=float(def_total), step=10.0)

# Build DataFrame with exact 19 approved features
input_dict = {
    "Gender": [gender],
    "Senior Citizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "Tenure Months": [int(tenure_months)],
    "Phone Service": [phone_service],
    "Multiple Lines": [multiple_lines],
    "Internet Service": [internet_service],
    "Online Security": [online_security],
    "Online Backup": [online_backup],
    "Device Protection": [device_protection],
    "Tech Support": [tech_support],
    "Streaming TV": [streaming_tv],
    "Streaming Movies": [streaming_movies],
    "Contract": [contract],
    "Paperless Billing": [paperless],
    "Payment Method": [payment_method],
    "Monthly Charges": [float(monthly_charges)],
    "Total Charges": [float(total_charges)],
}

input_df = pd.DataFrame(input_dict)

# Model Inference
try:
    churn_prob = float(pipeline.predict_proba(input_df)[0, 1])
except Exception as e:
    st.error(f"Inference error: {e}")
    st.stop()

# Analytical Risk Level
if churn_prob < 0.30:
    risk_level = "Low Risk"
    risk_class = "metric-value-low"
    risk_badge = "🟢 Low Risk"
elif churn_prob < 0.60:
    risk_level = "Medium Risk"
    risk_class = "metric-value-med"
    risk_badge = "🟡 Medium Risk"
else:
    risk_level = "High Risk"
    risk_class = "metric-value-high"
    risk_badge = "🔴 High Risk"

# Operational Threshold Prediction (0.60 threshold)
is_predicted_churn = churn_prob >= 0.60
predicted_status = "Likely to Churn" if is_predicted_churn else "Likely to Stay"
status_class = "metric-value-high" if is_predicted_churn else "metric-value-low"

# ----------------- Main Display -----------------

# 1. Metric Cards
st.subheader("🎯 Real-Time Churn Risk Assessment")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Churn Probability</div>
        <div class="{risk_class}">{churn_prob * 100:.1f}%</div>
        <div style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">Logistic Regression Model Output</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Assigned Risk Level</div>
        <div class="{risk_class}">{risk_level}</div>
        <div style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">Analytical Prioritization Tier</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Operational Classification</div>
        <div class="{status_class}">{predicted_status}</div>
        <div style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">Decision Threshold: 60.0%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Visual Probability Gauge / Bar
st.write(f"**Churn Risk Spectrum:** `{churn_prob*100:.1f}%`")
st.progress(min(max(churn_prob, 0.0), 1.0))

# 2. Risk Drivers & Retention Strategy
HIGH_CHARGES_CUTOFF = 89.85  # Data-driven 75th percentile from EDA

# Identify observable heuristic risk drivers
risk_drivers = []
if contract == "Month-to-month":
    risk_drivers.append("⚠️ **Month-to-Month Contract:** No long-term contractual commitment.")
if tenure_months <= 12:
    risk_drivers.append(r"⚠️ **Early Lifecycle Account (Tenure &le; 12 Months):** High-hazard introductory onboarding window.")
if monthly_charges >= HIGH_CHARGES_CUTOFF:
    risk_drivers.append(f"⚠️ **High Monthly Charges (&ge; ${HIGH_CHARGES_CUTOFF:.2f}):** Customer billing in top quartile (${monthly_charges:.2f}/mo).")
if tech_support == "No":
    risk_drivers.append("⚠️ **Lacks Technical Support:** No active technical support subscription.")
if online_security == "No":
    risk_drivers.append("⚠️ **Lacks Online Security:** Missing cybersecurity protection add-on.")
if payment_method == "Electronic check":
    risk_drivers.append("⚠️ **Manual Electronic Check Payment:** Non-automated payment channel associated with elevated churn.")
if internet_service == "Fiber optic":
    risk_drivers.append("⚠️ **Fiber Optic Broadband Tier:** Premium internet tier with heightened sensitivity.")

# Determine rule-based retention actions
retention_actions = []
if risk_level == "High Risk":
    if contract == "Month-to-month":
        retention_actions.append("💡 **Contract Migration:** Offer longer-term contract options / annual commitment incentives.")
    if tenure_months <= 12:
        retention_actions.append("💡 **Customer Success Check-In:** Conduct proactive onboarding check-in to ensure service satisfaction.")
    if monthly_charges >= HIGH_CHARGES_CUTOFF:
        retention_actions.append("💡 **Plan Optimization:** Review plan and bundle fit to provide competitive value.")
    if tech_support == "No":
        retention_actions.append("💡 **Tech Support Onboarding:** Proactively offer setup assistance and technical support access.")
    if online_security == "No":
        retention_actions.append("💡 **Security Bundle Offer:** Explain and bundle value-add cybersecurity options.")
    if payment_method == "Electronic check":
        retention_actions.append("💡 **Auto-Pay Setup:** Offer automated payment (bank transfer / credit card) incentive.")
    if not retention_actions:
        retention_actions.append("💡 **Priority Account Review:** Assign customer success representative for personal outreach.")
elif risk_level == "Medium Risk":
    retention_actions.append("💡 **Targeted Account Check-In:** Periodic relationship review and satisfaction survey.")
    if tech_support == "No" or online_security == "No":
        retention_actions.append("💡 **Value-Add Feature Awareness:** Highlight available security and support features.")
    if contract == "Month-to-month":
        retention_actions.append("💡 **Annual Option Presentation:** Introduce benefits of longer-term term agreements.")
else:
    retention_actions.append("✅ **Loyalty Program Engagement:** Standard service maintenance, appreciation rewards, and referral programs.")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔍 Observable Risk Drivers")
    if risk_drivers:
        for d in risk_drivers:
            st.markdown(f'<div class="driver-box">{d}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="neutral-box">✅ <b>No Elevated Risk Drivers:</b> Customer displays stable tenure and engagement patterns.</div>', unsafe_allow_html=True)

with col_right:
    st.subheader("💡 Recommended Retention Actions")
    for a in retention_actions:
        st.markdown(f'<div class="action-box">{a}</div>', unsafe_allow_html=True)
    st.caption("ℹ️ *Retention actions are heuristic business recommendations constructed from correlation patterns. They do not represent causal guarantees.*")

st.markdown("<br>", unsafe_allow_html=True)

# Details & Metadata Tabs
tab1, tab2 = st.tabs(["📄 Customer Feature Snapshot", "⚙️ Model & Methodology Details"])

with tab1:
    st.dataframe(input_df, use_container_width=True)

with tab2:
    st.markdown("""
    - **Model Architecture:** Balanced Logistic Regression Pipeline (`StandardScaler` + `OneHotEncoder(handle_unknown='ignore')` + `LogisticRegression(class_weight='balanced')`).
    - **Performance (Test Set):** Accuracy = 74.31%, Recall = 78.07%, Precision = 51.05%, F1-Score = 0.6173, ROC-AUC = 0.8489.
    - **Operational Threshold:** `0.60` (Maximizes F1-Score on test set while retaining 71.7% churn detection rate).
    - **Analytical Risk Tiers:**
      - **Low Risk:** Churn Probability < 30% (Observed Churn Rate: 4.91%).
      - **Medium Risk:** 30% <= Churn Probability < 60% (Observed Churn Rate: 22.76%).
      - **High Risk:** Churn Probability >= 60% (Observed Churn Rate: 58.10%).
    """)
