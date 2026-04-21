import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import streamlit.components.v1 as components

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Credit Guard | Professional Loan Underwriter",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# LOAD MODEL & ASSETS
# ===============================
@st.cache_resource
def load_assets():
    model_package = joblib.load("classification_model.pkl")
    scaler = joblib.load("scaler classification_model.pkl")
    return model_package["model"], model_package["threshold"], scaler

model, threshold, scaler = load_assets()

encoding_dict = {
    "person_home_ownership": {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3},
    "loan_intent": {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2, "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5},
    "loan_grade": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
    "cb_person_default_on_file": {"N": 0, "Y": 1}
}

# ===============================
# CUSTOM CSS — EMERALD FINANCE THEME
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

.stApp {
    background: #050807;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* ─── ANIMATED BACKGROUND ─── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(circle at 10% 10%, rgba(16, 185, 129, 0.07) 0%, transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(245, 158, 11, 0.05) 0%, transparent 40%);
    z-index: -1;
}

/* ─── HERO SECTION ─── */
.hero-container {
    text-align: left;
    padding: 1rem 0 3rem 0;
    animation: fadeIn 1s ease-out;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.main-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.sub-title {
    color: #5EEAD4;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 600;
}

/* ─── GLASS CARDS ─── */
div[data-testid="stVerticalBlock"] > div:has(div.card-glow) {
    background: rgba(10, 20, 18, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(16, 185, 129, 0.1);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 2rem;
}

/* ─── INPUT STYLING ─── */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
}

/* ─── BUTTON ─── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    padding: 1rem !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    transition: 0.3s all ease;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# UI COMPONENTS
# ===============================
st.markdown("""
<div class="hero-container">
    <div class="sub-title">Institutional Grade Risk Assessment</div>
    <div class="main-title">CreditGuard AI</div>
    <p style="color: #94A3B8; max-width: 600px;">
        Advanced machine learning protocol for real-time credit default probability analysis. 
        Input customer parameters below for a certified risk evaluation.
    </p>
</div>
""", unsafe_allow_html=True)

# Main Input Form
with st.container():
    st.markdown('<div class="card-glow"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 👤 Applicant")
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 0, 1000000, 50000, step=5000)
        emp_len = st.number_input("Work Experience (Years)", 0, 50, 5)
        home = st.selectbox("Residence Type", list(encoding_dict["person_home_ownership"].keys()))
        
    with col2:
        st.markdown("### 💰 Loan Terms")
        amount = st.number_input("Principal Amount ($)", 500, 500000, 10000, step=1000)
        intent = st.selectbox("Loan Purpose", list(encoding_dict["loan_intent"].keys()))
        grade = st.selectbox("Internal Credit Grade", list(encoding_dict["loan_grade"].keys()))
        rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 10.5, step=0.1)

    with col3:
        st.markdown("### 📜 History")
        prev_default = st.selectbox("Prior Defaults", list(encoding_dict["cb_person_default_on_file"].keys()))
        cred_len = st.number_input("Credit Age (Years)", 0, 50, 10)
        # Auto-calculate loan percent of income
        pct_income = amount / income if income > 0 else 0
        st.info(f"Debt-to-Income Ratio: {pct_income:.2%}")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Execute Risk Analysis")

# ===============================
# LOGIC & RESULTS
# ===============================
def build_result_html(prob, prediction, thresh):
    is_safe = prediction == 0
    color = "#10B981" if is_safe else "#F43F5E"
    status = "APPROVED / LOW RISK" if is_safe else "REJECTED / HIGH RISK"
    icon = "🛡️" if is_safe else "⚠️"
    
    return f"""
    <div style="font-family:'Inter',sans-serif; background:rgba(255,255,255,0.02); border:1px solid {color}44; border-radius:20px; padding:2rem; color:white; animation:fadeIn 0.6s ease-out;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem;">
            <div>
                <div style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.1em;">Decision Status</div>
                <div style="font-size:1.8rem; font-weight:700; color:{color};">{icon} {status}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase;">Model Confidence</div>
                <div style="font-size:1.8rem; font-weight:700; color:#F8FAFC;">{100 - (abs(prob-thresh)*100):.1f}%</div>
            </div>
        </div>
        
        <div style="background:rgba(0,0,0,0.2); height:12px; border-radius:10px; width:100%; margin-bottom:1rem; position:relative; overflow:hidden;">
            <div style="background:{color}; width:{prob*100}%; height:100%; border-radius:10px; transition:1s width ease-in-out;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#64748B;">
            <span>Probability of Default: {prob:.2%}</span>
            <span>Threshold: {thresh:.2f}</span>
        </div>
        
        <div style="margin-top:2rem; padding-top:1.5rem; border-top:1px solid rgba(255,255,255,0.05); font-size:0.9rem; line-height:1.6; color:#CBD5E1;">
            <b>Clinical Summary:</b> The underwriting model has processed the financial profile. 
            {"The applicant demonstrates strong fiscal stability with a risk profile within acceptable institutional bounds." if is_safe else "The analysis identifies critical risk markers in the debt-to-income ratio or credit history. Immediate rejection advised."}
        </div>
    </div>
    """

if predict_btn:
    # 1. Processing Animation
    with st.spinner("Analyzing Credit Risk..."):
        time.sleep(1.2) # Psychological delay for "heavy computing"
        
        # 2. Data Preparation
        encoded_inputs = {
            "person_age": age,
            "person_income": income,
            "person_emp_length": emp_len,
            "person_home_ownership": encoding_dict["person_home_ownership"][home],
            "loan_intent": encoding_dict["loan_intent"][intent],
            "loan_grade": encoding_dict["loan_grade"][grade],
            "loan_amnt": amount,
            "loan_int_rate": rate,
            "loan_percent_income": pct_income,
            "cb_person_default_on_file": encoding_dict["cb_person_default_on_file"][prev_default],
            "cb_person_cred_hist_length": cred_len,
        }
        
        input_df = pd.DataFrame([encoded_inputs])
        numeric_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # 3. Prediction
        prob_default = model.predict_proba(input_df)[:, 1][0]
        prediction = 1 if prob_default >= threshold else 0
        
        # 4. Display
        st.markdown("<br>", unsafe_allow_html=True)
        components.html(build_result_html(prob_default, prediction, threshold), height=400)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div style="text-align:center; margin-top:5rem; padding:2rem; opacity:0.5;">
    <div style="font-size:0.7rem; color:#94A3B8; letter-spacing:0.2em; text-transform:uppercase;">
        Secure Underwriting Protocol v4.0.1 • Scikit-Learn • XGBoost / Logistic Baseline
    </div>
</div>
""", unsafe_allow_html=True)
