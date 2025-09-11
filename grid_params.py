# Parameters for the first pipeline

# XGBoost
xgb_params = {
    "classifier__n_estimators": [200, 400, 600],   
    "classifier__max_depth": [3, 5, 7, 10],        
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__subsample": [0.6, 0.8, 1.0]       
}

# CatBoost
cat_params = {
    "classifier__iterations": [200, 400, 600],     
    "classifier__depth": [4, 6, 8, 10],            
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__l2_leaf_reg": [1, 3, 5, 7]        
}


#=================================================================

# Parameters for the second pipeline

xgb_params_ohe = {
    "classifier__n_estimators": [300, 600],   
    "classifier__max_depth": [4, 6],          
    "classifier__learning_rate": [0.05, 0.1], 
    "classifier__subsample": [0.8, 1.0]       
}

cat_params_ohe = {
    "classifier__iterations": [400, 600],     
    "classifier__depth": [6, 8],              
    "classifier__learning_rate": [0.05, 0.1], 
    "classifier__l2_leaf_reg": [3, 5]         
}

