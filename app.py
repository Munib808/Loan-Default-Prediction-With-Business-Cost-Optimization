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
# CUSTOM CSS — PROFESSIONAL ENGINEERING THEME
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #060B0A;
    --panel: #0D1512;
    --panel-border: rgba(94, 234, 212, 0.14);
    --accent: #10B981;
    --accent-bright: #34D399;
    --accent-soft: rgba(16, 185, 129, 0.12);
    --danger: #F43F5E;
    --text-primary: #F1F5F4;
    --text-secondary: #9CA9A5;
    --text-muted: #6B7A76;
    --mono: 'JetBrains Mono', monospace;
}

.stApp {
    background: var(--bg);
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1220px; }

/* ─── BACKGROUND GLOW ─── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(circle at 8% 8%, rgba(16, 185, 129, 0.09) 0%, transparent 42%),
        radial-gradient(circle at 92% 85%, rgba(52, 211, 153, 0.06) 0%, transparent 42%);
    z-index: -1;
}

/* ─── HERO ─── */
.hero-container { padding: 0.5rem 0 2rem 0; animation: fadeIn 0.7s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

.status-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--accent-soft); border: 1px solid rgba(16,185,129,0.35);
    color: var(--accent-bright); font-family: var(--mono); font-size: 0.72rem;
    font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.35rem 0.9rem; border-radius: 999px; margin-bottom: 1rem;
}
.status-pill .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--accent-bright);
    box-shadow: 0 0 8px var(--accent-bright);
}

.main-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.02em;
    margin-bottom: 0.4rem; line-height: 1.1;
}
.sub-title {
    color: var(--text-secondary);
    font-size: 1.02rem; max-width: 640px; line-height: 1.6;
}

/* ─── SECTION HEADERS INSIDE CARDS ─── */
.section-label {
    font-family: var(--mono); font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent-bright); margin-bottom: 1.1rem;
    display: flex; align-items: center; gap: 0.5rem;
    border-bottom: 1px solid var(--panel-border); padding-bottom: 0.6rem;
}

/* ─── GLASS CARDS ─── */
div[data-testid="stVerticalBlock"] > div:has(div.card-glow) {
    background: linear-gradient(180deg, rgba(16,26,22,0.75), rgba(8,14,12,0.75));
    backdrop-filter: blur(14px);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 2rem 2.2rem 1.6rem 2.2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* ─── LABELS (this is what was invisible before) ─── */
label, .stNumberInput label, .stSelectbox label, .stTextInput label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
}
label p { color: var(--text-secondary) !important; }

/* ─── INPUT FIELDS ─── */
.stNumberInput input, .stTextInput input {
    background-color: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 0.8rem !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
.stNumberInput button {
    background-color: rgba(255,255,255,0.03) !important;
    border-color: var(--panel-border) !important;
}
.stNumberInput svg { fill: var(--text-secondary) !important; }

/* ─── SELECTBOX (closed state) ─── */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.stSelectbox div[data-baseweb="select"] span {
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
}
.stSelectbox svg { fill: var(--text-secondary) !important; }

/* ─── SELECTBOX DROPDOWN MENU (this was invisible: dark text on dark bg) ─── */
ul[data-baseweb="menu"], div[data-baseweb="popover"] li {
    background-color: #0D1512 !important;
}
li[role="option"] {
    background-color: #0D1512 !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    font-size: 0.9rem !important;
}
li[role="option"]:hover, li[aria-selected="true"] {
    background-color: var(--accent-soft) !important;
    color: var(--accent-bright) !important;
}

/* ─── ST.INFO BOX ─── */
div[data-testid="stAlertContainer"] {
    background-color: rgba(16, 185, 129, 0.08) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    border-radius: 10px !important;
}
div[data-testid="stAlertContainer"] p {
    color: var(--accent-bright) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}

/* ─── BUTTON ─── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: #04120C !important;
    border: none !important;
    padding: 0.95rem !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    transition: 0.25s all ease;
    box-shadow: 0 4px 18px rgba(16, 185, 129, 0.25) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(16, 185, 129, 0.4) !important;
}
div.stButton > button:active { transform: translateY(0); }

/* ─── SPINNER TEXT ─── */
.stSpinner p { color: var(--text-secondary) !important; font-family: var(--mono) !important; }

/* ─── FOOTER ─── */
.app-footer {
    text-align: center; margin-top: 4rem; padding: 1.6rem;
    border-top: 1px solid var(--panel-border);
}
.app-footer span {
    font-family: var(--mono); font-size: 0.7rem; color: var(--text-muted);
    letter-spacing: 0.18em; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HERO
# ===============================
st.markdown("""
<div class="hero-container">
    <div class="status-pill"><span class="dot"></span> Model Online · v4.0.1</div>
    <div class="main-title">CreditGuard AI</div>
    <p class="sub-title">
        Institutional-grade machine learning protocol for real-time credit default probability analysis.
        Enter the applicant's financial profile below to generate a certified underwriting decision.
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# MAIN INPUT FORM
# ===============================
with st.container():
    st.markdown('<div class="card-glow"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-label">👤&nbsp; Applicant Profile</div>', unsafe_allow_html=True)
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 0, 1000000, 50000, step=5000)
        emp_len = st.number_input("Work Experience (Years)", 0, 50, 5)
        home = st.selectbox("Residence Type", list(encoding_dict["person_home_ownership"].keys()))

    with col2:
        st.markdown('<div class="section-label">💰&nbsp; Loan Terms</div>', unsafe_allow_html=True)
        amount = st.number_input("Principal Amount ($)", 500, 500000, 10000, step=1000)
        intent = st.selectbox("Loan Purpose", list(encoding_dict["loan_intent"].keys()))
        grade = st.selectbox("Internal Credit Grade", list(encoding_dict["loan_grade"].keys()))
        rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 10.5, step=0.1)

    with col3:
        st.markdown('<div class="section-label">📜&nbsp; Credit History</div>', unsafe_allow_html=True)
        prev_default = st.selectbox("Prior Defaults", list(encoding_dict["cb_person_default_on_file"].keys()))
        cred_len = st.number_input("Credit Age (Years)", 0, 50, 10)
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
    status = "APPROVED · LOW RISK" if is_safe else "REJECTED · HIGH RISK"
    icon = "🛡️" if is_safe else "⚠️"

    return f"""
    <div style="font-family:'Inter',sans-serif; background:rgba(255,255,255,0.02); border:1px solid {color}55; border-radius:18px; padding:2rem; color:#F1F5F4; animation:fadeIn 0.6s ease-out;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.8rem; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#9CA9A5; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.3rem;">Decision Status</div>
                <div style="font-size:1.7rem; font-weight:700; color:{color};">{icon} {status}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#9CA9A5; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.3rem;">Model Confidence</div>
                <div style="font-size:1.7rem; font-weight:700; color:#F1F5F4;">{100 - (abs(prob-thresh)*100):.1f}%</div>
            </div>
        </div>

        <div style="background:rgba(0,0,0,0.3); height:10px; border-radius:8px; width:100%; margin-bottom:0.8rem; position:relative; overflow:hidden;">
            <div style="background:{color}; width:{prob*100}%; height:100%; border-radius:8px; transition:1s width ease-in-out;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#6B7A76;">
            <span>Probability of Default: {prob:.2%}</span>
            <span>Decision Threshold: {thresh:.2f}</span>
        </div>

        <div style="margin-top:1.8rem; padding-top:1.4rem; border-top:1px solid rgba(255,255,255,0.06); font-size:0.88rem; line-height:1.6; color:#CBD5E1;">
            <b style="color:#F1F5F4;">Model Summary:</b>
            {"The applicant demonstrates strong fiscal stability with a risk profile within acceptable institutional bounds." if is_safe else "The analysis identifies elevated risk markers in the debt-to-income ratio or credit history. Manual review recommended."}
        </div>
    </div>
    """

if predict_btn:
    with st.spinner("Analyzing Credit Risk..."):
        time.sleep(1.2)

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

        prob_default = model.predict_proba(input_df)[:, 1][0]
        prediction = 1 if prob_default >= threshold else 0

        st.markdown("<br>", unsafe_allow_html=True)
        components.html(build_result_html(prob_default, prediction, threshold), height=380)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="app-footer">
    <span>Secure Underwriting Protocol v4.0.1 · Scikit-Learn · XGBoost / Logistic Baseline</span>
</div>
""", unsafe_allow_html=True)
