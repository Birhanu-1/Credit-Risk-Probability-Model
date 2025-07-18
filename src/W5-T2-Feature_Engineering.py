# Import Labraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOE
# Extracts temporal features from the 'TransactionStartTime
class TransactionTimeFeatures(BaseEstimator, TransformerMixin):
   
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        X['TransactionHour'] = X[self.datetime_col].dt.hour
        X['TransactionDay'] = X[self.datetime_col].dt.day
        X['TransactionMonth'] = X[self.datetime_col].dt.month
        X['TransactionYear'] = X[self.datetime_col].dt.year
        return X.drop(columns=[self.datetime_col])
# Computes aggregate transaction statistics
class CustomerAggregates(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg = X.groupby('CustomerId')['Amount'].agg([
            ('TotalTransactionAmount', 'sum'),
            ('AverageTransactionAmount', 'mean'),
            ('TransactionCount', 'count'),
            ('StdDevTransactionAmount', 'std')
        ]).reset_index()
        return X.merge(agg, on='CustomerId', how='left')
# Main Feature Engineering Pipeline
def build_feature_pipeline():
    numerical_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth',
                          'TransactionYear', 'TotalTransactionAmount', 'AverageTransactionAmount',
                          'TransactionCount', 'StdDevTransactionAmount']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
                            'ChannelId', 'PricingStrategy']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    pipeline = Pipeline([
        ('time_features', TransactionTimeFeatures(datetime_col='TransactionStartTime')),
        ('aggregate_features', CustomerAggregates()),
        ('preprocessor', preprocessor)
    ])

    return pipeline

# Sample Usage with Local Dataset
def run_local_pipeline(filepath):
    """
    Load local dataset and run the feature engineering pipeline.
    """
    df = pd.read_csv(filepath)
    pipeline = build_feature_pipeline()
    transformed_data = pipeline.fit_transform(df)
    return transformed_data

transformed = run_local_pipeline("../data/data.csv")
print(transformed.shape)


