# Loan Default Prediction with Business Cost Optimization

##  Project Overview
This project focuses on building a machine learning system to predict loan defaults while optimizing for **business costs**.  
Unlike traditional classification that uses a `0.5 threshold`, this implementation finds the **optimal threshold** that minimizes financial losses by balancing the costs of **false positives (FP)** and **false negatives (FN)**.

---

##  Business Problem

In lending, different types of errors have different financial impacts:

- **False Positive (FP)**: Approving a loan that will default  
  - Cost = Loss of principal + interest = **$5,000**

- **False Negative (FN)**: Rejecting a loan that would have been paid back  
  - Cost = Lost interest income = **$15,000**

**Objective:** Minimize  
\[
\text{Total Cost} = (FP \times 5000) + (FN \times 15000)
\]

---

##  Dataset

- **Source:** `credit_risk_dataset.csv`  
- **Size:** 32,581 loan applications with 12 features  

### Features
- **Demographic:** `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`
- **Loan characteristics:** `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`
- **Credit history:** `cb_person_default_on_file`, `cb_person_cred_hist_length`
- **Target:** `loan_status`  
  - `0 = non-default`  
  - `1 = default`

### Class Distribution
- **Non-Default Cases:** 25,473 (78.2%)  
- **Default Cases:** 7,108 (21.8%)

---

##  Technical Implementation

### 🔹 Data Preprocessing
- Missing values handled  
  - Median imputation for `person_emp_length`  
  - Dropped NA for `loan_int_rate`  
- Outliers removed (`age < 100`, `employment length < 50 years`)  
- Categorical variables encoded (`Label Encoding`)  
- Numerical features scaled (`StandardScaler`)  

### 🔹 Models Evaluated
- Logistic Regression (Baseline)  
- Random Forest (GridSearch optimized)  
- CatBoost (GridSearch optimized) → **Best Performing**  

### 🔹 Optimal Threshold Finding
- Business costs calculated across thresholds  
- **Optimal threshold = 0.231** (vs default 0.5)  
- **Cost reduction with optimal threshold: $725000 (8.45%)**

---

##  Key Insights
- Loan **grade** strongly correlates with default rates — higher grades (A) have lower defaults  
- Loan **percentage of income** is critical — values above 0.5 show significantly higher default rates  
- **Interest rates** are highly predictive — higher rates correlate with higher default probabilities  
- **Top features:**  
  - `loan_percent_income`  
  - `loan_int_rate`  
  - `loan_grade`  
  - `person_home_ownership`  

---

##  Files
- `Loan Default With Business Cost Optimization(1).ipynb` → Complete analysis notebook  
- `classification_model.pkl` → Trained CatBoost model with optimal threshold  
- `scaler_classification_model.pkl` → Feature scaler  
- `credit_risk_dataset.csv` → Original dataset  
- `requirements.txt` → Python dependencies  

---

##  Results (CatBoost Model)
- **Accuracy:** 92%   

---

##  Live App
You can try the deployed app here:  
https://loan-default-prediction-with-business-cost-optimization-mgaufu.streamlit.app/
