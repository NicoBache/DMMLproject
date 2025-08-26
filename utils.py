import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ===========================
# Function to plot importances
# ===========================
def plot_feature_importances(model, feature_names, model_name, top_n=15):
    """
    Plot top_n feature importances for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif model.__class__.__name__ == "CatBoostClassifier":
        importances = model.get_feature_importance()
    else:
        raise ValueError(f"Model {model_name} does not provide feature importances.")
    
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, top_n * 0.4 + 1))
    sns.barplot(x="importance", y="feature", data=df_imp, palette="viridis")
    plt.title(f"{model_name} - Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()


# ===========================
# Function to train & plot importances for all models
# ===========================
def feature_importance_all_models(X, y, numerical_features, categorical_features, preprocessor, use_smote=False, top_n=15, title_suffix=""):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1, class_weight="balanced"),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=1),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=1, class_weight="balanced"),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42, auto_class_weights="Balanced")
    }

    # Fit preprocessor per ottenere le colonne trasformate
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)

    # Indici categoriche coerenti con i dati preprocessati
    cat_indices = [X_transformed.columns.get_loc(c) for c in categorical_features]

    print("\n=== Debug indici categoriche ===")
    for c in categorical_features:
        print(f"{c:30} -> indice {X_transformed.columns.get_loc(c)}")
    print("Indici finali:", cat_indices)

    for name, clf in models.items():
        if use_smote:
            pipe = ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote", SMOTENC(categorical_features=cat_indices, random_state=42)),
                ("classifier", clf)
            ])
        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])

        pipe.fit(X, y)

        # Feature names: numeriche + categoriche (ordine coerente al preprocessor)
        all_features = numerical_features + categorical_features

        # Estrai il classificatore
        model_inside = pipe.named_steps["classifier"]

        # Plot importances
        plot_feature_importances(model_inside, all_features, f"{name} {title_suffix}", top_n=top_n)
