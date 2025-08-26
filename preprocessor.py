from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # definisco i transformer standard sklearn
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=int
        )
        
    def fit(self, X, y=None):
        # fit numeriche
        self.num_imputer.fit(X[self.numerical_features])
        X_num = self.num_imputer.transform(X[self.numerical_features])
        self.scaler.fit(X_num)
        
        # fit categoriche
        X_cat = self.cat_imputer.fit_transform(X[self.categorical_features])
        self.encoder.fit(X_cat)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # trasforma numeriche
        X_num = self.num_imputer.transform(X[self.numerical_features])
        X_num = self.scaler.transform(X_num)
        df_num = pd.DataFrame(X_num, columns=self.numerical_features, index=X.index)
        
        # trasforma categoriche
        X_cat = self.cat_imputer.transform(X[self.categorical_features])
        X_cat = self.encoder.transform(X_cat)
        df_cat = pd.DataFrame(X_cat, columns=self.categorical_features, index=X.index)
        
        # concatena in un unico DataFrame
        X_out = pd.concat([df_num, df_cat], axis=1)
        return X_out
