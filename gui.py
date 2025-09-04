import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Carica la pipeline salvata
pipeline = joblib.load("Models/best_pipeline.joblib")

st.title("Credit Risk Prediction — Demo GUI")

# AGGIUNGI CAMPI DELLE VARIE ROBE CATEGORICHE #######################################

# --- Input utente per tutte le feature originali ---
person_age = st.number_input("Età", min_value=18, max_value=99, value=30)
person_gender = st.selectbox("Genere", ["male", "female"])
person_education = st.selectbox("Titolo di studio", ["High School", "Bachelor", "Master", "PhD", "Other"])
person_income = st.number_input("Reddito annuale", min_value=0, value=30000)
person_emp_exp = st.number_input("Esperienza lavorativa (anni)", min_value=0, value=5)
person_home_ownership = st.selectbox("Proprietà casa", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Importo prestito", min_value=500, value=5000)
loan_intent = st.selectbox("Motivo prestito", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION", "OTHER"])
loan_int_rate = st.number_input("Tasso interesse (%)", min_value=0.0, max_value=30.0, value=12.0)
loan_percent_income = st.number_input("Percentuale prestito su reddito", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Lunghezza storia creditizia (anni)", min_value=0, value=3)
credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=650)

# --- Feature engineering identico al notebook ---
input_dict = {
    "person_age": person_age,
    "person_gender": person_gender,
    "person_education": person_education,
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": person_home_ownership,
    "loan_amnt": loan_amnt,
    "loan_intent": loan_intent,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
}

# Age bin
input_dict["person_age_bin"] = pd.cut(
    [person_age], bins=[0, 25, 35, 50, np.inf], labels=['<25', '25-35', '35-50', '50+']
)[0]

# income_to_loan
input_dict["income_to_loan"] = person_income / loan_amnt if loan_amnt else np.nan

# emp_exp_x_age
input_dict["emp_exp_x_age"] = person_emp_exp / person_age if person_age else np.nan

# loan_over_score
input_dict["loan_over_score"] = loan_amnt / credit_score if credit_score else np.nan

# loan_int_rate_bin
input_dict["loan_int_rate_bin"] = pd.cut(
    [loan_int_rate], bins=[-np.inf, 10, 15, 20, np.inf], labels=['<10%', '10-15%', '15-20%', '20%+']
)[0]

# --- Costruisci DataFrame per la pipeline ---
input_df = pd.DataFrame([input_dict])

# --- Predizione ---
if st.button("Predici rischio"):
    pred = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]
    st.success(f"Predizione: {'Default' if pred == 1 else 'No Default'}")
    st.info(f"Probabilità di default: {proba:.2%}")