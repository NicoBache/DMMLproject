# === PREPROCESSOR OHE MINIMALE (clone-safe) ===
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


categorical_features_ohe = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "person_age_bin",
    "loan_int_rate_bin"
]

numerical_features_ohe = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "income_to_loan",
    "emp_exp_x_age",
    "loan_over_score"
]



ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor_ohe = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical_features_ohe),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]), categorical_features_ohe),
    ],
    remainder="drop",
)
