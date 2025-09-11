# run with: 
# streamlit run gui.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
#  Load the trained pipeline
# -------------------------------
pipeline = joblib.load("Models/best_pipeline.joblib")

st.title("Credit Risk Prediction")

# -------------------------------
# Inputs – keep labels identical to training CSV
# -------------------------------

# CATEGORICALS: use the exact set seen in training
person_gender = st.selectbox("Gender", ["female", "male"])

person_education = st.selectbox(
    "Education",
    ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
)

person_home_ownership = st.selectbox(
    "Home ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_intent = st.selectbox(
    "Loan intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

# for numerics using ranges seen in CSV
person_age = st.number_input("Age", min_value=20, max_value=144, value=30)
person_income = st.number_input("Annual income", min_value=8000, max_value=7_200_766, value=30_000)
person_emp_exp = st.number_input("Work experience (years)", min_value=0, max_value=125, value=5)
loan_amnt = st.number_input("Loan amount", min_value=500, max_value=35_000, value=5_000)
loan_int_rate = st.number_input("Interest rate (%)", min_value=5.42, max_value=20.0, value=12.0, step=0.01)
#loan_percent_income = st.number_input("Loan / income ratio", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
cb_person_cred_hist_length = st.number_input("Credit history length (years)", min_value=2, max_value=30, value=3)
credit_score = st.number_input("Credit score", min_value=390, max_value=850, value=650)

# -------------------------------
# Feature engineering
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
    "loan_int_rate": float(loan_int_rate),      
    #"loan_percent_income": float(loan_percent_income),
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
}

# Loan-to-income ratio, original feature which can be computed from income and amount
input_dict["loan_percent_income"] = (loan_amnt / person_income)


#=========================
# Engineered features
#=========================
# Age bin 
input_dict["person_age_bin"] = pd.cut(
    [person_age], bins=[0, 25, 35, 50, np.inf], labels=['<25', '25-35', '35-50', '50+']
)[0]

# Income-to-loan ratio
input_dict["income_to_loan"] = (person_income / loan_amnt) if loan_amnt else np.nan

# Work experience relative to age
input_dict["emp_exp_x_age"] = (person_emp_exp / person_age) if person_age else np.nan

# Loan amount divided by credit score
input_dict["loan_over_score"] = (loan_amnt / credit_score) if credit_score else np.nan

# Interest rate bin 
input_dict["loan_int_rate_bin"] = pd.cut(
    [loan_int_rate], bins=[-np.inf, 10, 15, 20, np.inf], labels=['<10%', '10-15%', '15-20%', '20%+']
)[0]

# Single-row DataFrame for the pipeline
input_df = pd.DataFrame([input_dict])

# -------------------------------
# 3) Predict — the pipeline preprocessor will:
#    - impute numerics (median) and scale them,
#    - impute categoricals (most_frequent),
#    - encode categoricals with OrdinalEncoder,
#    then pass the transformed features to the classifier.
# -------------------------------
if st.button("Predict risk"):
    # Predicted class: 1 = Default, 0 = No Default
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

