
# Random Forest
rf_params = {
    "classifier__n_estimators": [200, 400, 600],   # più alberi → più stabilità, ma più tempo   4
    "classifier__max_depth": [None, 10, 20],   # limite profondità      1
    "classifier__min_samples_split": [2, 5, 10],   # split più restrittivi  3
    "classifier__min_samples_leaf": [1, 2, 4]      # numero minimo di campioni per foglia   2
}

# XGBoost
xgb_params = {
    "classifier__n_estimators": [200, 400, 600],   # numero di boosting rounds
    "classifier__max_depth": [3, 5, 7, 10],        # profondità degli alberi
    "classifier__learning_rate": [0.01, 0.05, 0.1],# tasso di apprendimento
    "classifier__subsample": [0.6, 0.8, 1.0]       # frazione di campioni per boosting round
}

# CatBoost
cat_params = {
    "classifier__iterations": [200, 400, 600],     # numero di boosting iterations
    "classifier__depth": [4, 6, 8, 10],            # profondità degli alberi
    "classifier__learning_rate": [0.01, 0.05, 0.1],# tasso di apprendimento
    "classifier__l2_leaf_reg": [1, 3, 5, 7]        # regolarizzazione L2
}


# Parameters for OHE pipeline

rf_params_ohe = {
    "classifier__n_estimators": [200, 400],   # taglia via 600
    "classifier__max_depth": [None, 12],      # meno valori di profondità
    "classifier__min_samples_split": [2, 5],  # due soli valori
    "classifier__min_samples_leaf": [1, 2]    # idem
}

xgb_params_ohe = {
    "classifier__n_estimators": [300, 600],   # due valori, non tre
    "classifier__max_depth": [4, 6],          # tagliando 3 e 10
    "classifier__learning_rate": [0.05, 0.1], # eviti l’1e-2 troppo lento
    "classifier__subsample": [0.8, 1.0]       # due valori
}

cat_params_ohe = {
    "classifier__iterations": [400, 600],     # via il 200
    "classifier__depth": [6, 8],              # valori medi
    "classifier__learning_rate": [0.05, 0.1], # due valori
    "classifier__l2_leaf_reg": [3, 5]         # due valori
}





"""
rf_params = {
    "classifier__n_estimators": [100, 300, 500, 800],  
    # Number of trees in the forest (higher = more stable but slower)

    "classifier__max_depth": [None, 10, 20, 30],  
    # Maximum depth of each tree (None = expand until all leaves are pure)

    "classifier__min_samples_split": [2, 5, 10],  
    # Minimum number of samples required to split an internal node
    # (higher = more regularization, prevents overfitting)

    "classifier__min_samples_leaf": [1, 2, 4]  
    # Minimum number of samples required to be at a leaf node
    # (higher = smoother decision boundaries)
}


xgb_params = {
    "classifier__n_estimators": [200, 500, 800],  
    # Number of boosting rounds (trees)

    "classifier__max_depth": [3, 5, 7, 10],  
    # Maximum depth of trees (controls model complexity)

    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],  
    # Step size shrinkage (smaller = slower learning but more accurate)

    "classifier__subsample": [0.6, 0.8, 1.0]  
    # Fraction of samples used for fitting each tree (like bagging, reduces overfitting)
}


cat_params = {
    "classifier__iterations": [300, 500, 800],  
    # Number of boosting iterations (similar to n_estimators)

    "classifier__depth": [4, 6, 8, 10],  
    # Depth of each tree (higher = more complex model)

    "classifier__learning_rate": [0.01, 0.05, 0.1],  
    # Step size shrinkage (trade-off between speed and accuracy)

    "classifier__l2_leaf_reg": [1, 3, 5, 7]  
    # L2 regularization term on leaf weights (higher = more regularization, less overfitting)
}

"""