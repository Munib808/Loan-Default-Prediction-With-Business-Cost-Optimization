import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load model and scaler
# -------------------------------
model_package = joblib.load("classification_model.pkl")
model = model_package["model"]
threshold = model_package["threshold"]

scaler = joblib.load("scaler classification_model.pkl")

# -------------------------------
# Encoding dictionaries
# -------------------------------
encoding_dict = {
    "person_home_ownership": {
        "MORTGAGE": 0,
        "OTHER": 1,
        "OWN": 2,
        "RENT": 3
    },
    "loan_intent": {
        "DEBTCONSOLIDATION": 0,
        "EDUCATION": 1,
        "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3,
        "PERSONAL": 4,
        "VENTURE": 5
    },
    "loan_grade": {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6
    },
    "cb_person_default_on_file": {
        "N": 0,
        "Y": 1
    }
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Loan Default Prediction App")

st.write("Fill in the details below and click *Predict* to see if the loan is likely to default.")

# Collect inputs
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=0, step=1000, value=50000)
person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=60, value=5)

person_home_ownership = st.selectbox("Home Ownership", list(encoding_dict["person_home_ownership"].keys()))
loan_intent = st.selectbox("Loan Intent", list(encoding_dict["loan_intent"].keys()))
loan_grade = st.selectbox("Loan Grade", list(encoding_dict["loan_grade"].keys()))

loan_amnt = st.number_input("Loan Amount", min_value=500, step=500, value=10000)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, step=0.1, value=10.0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, step=0.01, value=0.2)

cb_person_default_on_file = st.selectbox("Previously Defaulted?", list(encoding_dict["cb_person_default_on_file"].keys()))
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=60, value=10)

# -------------------------------
# Prediction Logic
# -------------------------------
if st.button("Predict"):
    # Encode categorical features
    encoded_inputs = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "person_home_ownership": encoding_dict["person_home_ownership"][person_home_ownership],
        "loan_intent": encoding_dict["loan_intent"][loan_intent],
        "loan_grade": encoding_dict["loan_grade"][loan_grade],
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": encoding_dict["cb_person_default_on_file"][cb_person_default_on_file],
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([encoded_inputs])

    # Scale numerical columns (fit was already done during training)
    numeric_cols = ["person_age", "person_income", "person_emp_length",
                    "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict probabilities
    prob_default = model.predict_proba(input_df)[:, 1][0]

    # Apply threshold
    prediction = 1 if prob_default >= threshold else 0

    # Display results
    st.subheader("🔎 Prediction Result")
    st.write(f"**Probability of Default:** {prob_default:.3f}")
    st.write(f"**Threshold Used:** {threshold}")
    if prediction == 1:
        st.error("⚠️ High Risk: Loan is likely to default")
    else:
        st.success("✅ Low Risk: Loan is unlikely to default")
