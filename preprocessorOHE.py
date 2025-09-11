from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def make_preprocessor_ohe(numerical_features, categorical_features):

    # OneHotEncoder for categorical features
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    return ColumnTransformer(
        transformers=[
            # pipeline for numerical features
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numerical_features),
            # pipeline for categorical features
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe),
            ]), categorical_features),
        ],
        remainder="drop", 
    )