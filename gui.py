# app_streamlit_ordinal_smote.py
# --------------------------------
# Streamlit GUI for a pipeline trained with:
# - Custom preprocessor using OrdinalEncoder (handle_unknown='use_encoded_value', unknown_value=-1)
# - SMOTENC in training (note: SMOTE/SMOTENC is ignored at predict time)
#
# The GUI:
# 1) Collects inputs using the SAME categorical labels as in training/CSV
# 2) Builds the same engineered features used in your notebook
# 3) Calls pipeline.predict / predict_proba
#
# NOTE: We intentionally DO NOT include 'previous_loan_defaults_on_file' because you didn't use it.

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# 0) Load the trained pipeline
# -------------------------------
# This should be the exact pipeline you saved (with preprocessor + SMOTENC + classifier).
# SMOTENC is only active during fit, not during predict, so it's safe to deploy.
pipeline = joblib.load("Models/best_pipeline.joblib")

st.title("Credit Risk Prediction — OrdinalEncoder + SMOTENC (Demo)")

# -------------------------------
# 1) Inputs – keep labels identical to training CSV
#    (OrdinalEncoder relies on consistent string labels; unknowns map to -1 per your settings)
# -------------------------------

# CATEGORICALS: use the exact set seen in training (case-sensitive)
person_gender = st.selectbox("Gender", ["female", "male"])

# Education: dataset uses 'Doctorate' (not 'PhD') and includes 'Associate'
person_education = st.selectbox(
    "Education",
    ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
)

# Home ownership: dataset has these four values (uppercase)
person_home_ownership = st.selectbox(
    "Home ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

# Loan intent: dataset has these six values (no 'OTHER')
loan_intent = st.selectbox(
    "Loan intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

# NUMERICALS: ranges aligned to your CSV (kept wide but realistic)
# (Min–max observed: age 20–144, income 8k–7.2M, emp_exp 0–125, loan 500–35000,
#  rate 5.42–20.0, loan_percent_income 0–0.66, credit history 2–30, score 390–850)

person_age = st.number_input("Age", min_value=20, max_value=144, value=30)
person_income = st.number_input("Annual income", min_value=8000, max_value=7_200_766, value=30_000)
person_emp_exp = st.number_input("Work experience (years)", min_value=0, max_value=125, value=5)
loan_amnt = st.number_input("Loan amount", min_value=500, max_value=35_000, value=5_000)
loan_int_rate = st.number_input("Interest rate (%)", min_value=5.42, max_value=20.0, value=12.0, step=0.01)
loan_percent_income = st.number_input("Loan / income ratio", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
cb_person_cred_hist_length = st.number_input("Credit history length (years)", min_value=2, max_value=30, value=3)
credit_score = st.number_input("Credit score", min_value=390, max_value=850, value=650)

# -------------------------------
# 2) Feature engineering — mirror your notebook logic
#    (These derived features must match names/dtypes used at training time)
# -------------------------------
input_dict = {
    # original features
    "person_age": person_age,
    "person_gender": person_gender,
    "person_education": person_education,
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": person_home_ownership,
    "loan_amnt": loan_amnt,
    "loan_intent": loan_intent,
    "loan_int_rate": float(loan_int_rate),      # ensure float dtype
    "loan_percent_income": float(loan_percent_income),
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
}

# Age bin — categorical buckets consistent with your notebook
# This creates a single label like '<25', '25-35', '35-50', '50+'
input_dict["person_age_bin"] = pd.cut(
    [person_age], bins=[0, 25, 35, 50, np.inf], labels=['<25', '25-35', '35-50', '50+']
)[0]

# Income-to-loan ratio (numeric)
input_dict["income_to_loan"] = (person_income / loan_amnt) if loan_amnt else np.nan

# Work experience relative to age (numeric)
input_dict["emp_exp_x_age"] = (person_emp_exp / person_age) if person_age else np.nan

# Loan amount divided by credit score (numeric)
input_dict["loan_over_score"] = (loan_amnt / credit_score) if credit_score else np.nan

# Interest rate bin — categorical buckets consistent with your notebook
# Up to 20% falls into the '20%+' bucket
input_dict["loan_int_rate_bin"] = pd.cut(
    [loan_int_rate], bins=[-np.inf, 10, 15, 20, np.inf], labels=['<10%', '10-15%', '15-20%', '20%+']
)[0]

# Build a single-row DataFrame for the pipeline
input_df = pd.DataFrame([input_dict])

# -------------------------------
# 3) Predict — the pipeline preprocessor will:
#    - impute numerics (median) and scale them,
#    - impute categoricals (most_frequent),
#    - encode categoricals with OrdinalEncoder (unknowns -> -1),
#    then pass the transformed features to the classifier.
# -------------------------------
if st.button("Predict risk"):
    # Predicted class: 1 = Default, 0 = No Default (as per your dataset)
    pred = pipeline.predict(input_df)[0]

    # Predicted probability for the positive class (Default=1)
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(input_df)[0][1]
    else:
        # Fallback to decision_function normalized in [0,1] if needed
        scores = pipeline.decision_function(input_df)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        proba = float(proba[0])

    # Display results
    st.success(f"Prediction: {'Default' if pred == 1 else 'No Default'}")
    st.info(f"Probability of default: {proba:.2%}")

