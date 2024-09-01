
""""
Aim to include most of preprocessing steps into pipeline as much as possible
To be able to apply to new data with a single transform call
"""""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from pmdarima.preprocessing import impute_seasonal

"""
In order will have the following preprocessing for pipelining:
1. Date-time columns conversion
2. Interpolater (Will not perfectly change all Nulls as we saw in tests. E.g. some Nulls at the start of a group)
3. Imputers. Using MinMaxScaler
"""

##########################################################

# Function to remove rows with too many missing values
def drop_rows_with_missing_values(X, y, threshold=None):
    if threshold is not None:
        # Calculate the number of missing values in each row
        missing_counts = X.isnull().sum(axis=1)

        # Filter rows based on the threshold
        rows_to_keep = missing_counts <= threshold

        # Apply the filter to both X and y
        X_cleaned = X[rows_to_keep]
        y_cleaned = y[rows_to_keep]

        return X_cleaned, y_cleaned
    else:
        # If no threshold is provided, return X and y as is
        return X, y

##########################################################

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='Date'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['Date_DT'] = pd.to_datetime(X_[self.date_column])
        X_['DayOfWeek'] = X_['Date_DT'].dt.dayofweek
        X_['IsWeekend'] = X_['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        return X_

##########################################################

class InterpolationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_interpolate=None, columns_to_drop=None, columns_to_keep=None):
        self.columns_to_interpolate = columns_to_interpolate
        self.columns_to_drop = columns_to_drop or []
        self.columns_to_keep = columns_to_keep
        self.columns_with_missing = None
        self.missing_percentages = None

    def fit(self, X, y=None):
        # Store the column names from the training data, excluding columns_to_drop
        self.train_columns = [col for col in X.columns if col not in self.columns_to_drop]

        # Determine which columns have missing values and should be interpolated
        self.columns_with_missing = [col for col in self.train_columns if X[col].isnull().any()]

        if self.columns_to_interpolate is None:
            self.columns_to_interpolate_ = self.columns_with_missing
        else:
            self.columns_to_interpolate_ = [col for col in self.columns_to_interpolate if
                                            col in self.columns_with_missing]

        # Calculate the percentage of missing values in each column
        self.missing_percentages = X[self.columns_with_missing].isnull().mean()

        return self

    def transform(self, X):
        X_ = X.copy()

        # Determine which columns are actually present in X
        available_columns = X_.columns.tolist()

        # Filter columns_to_interpolate based on available columns
        columns_to_interpolate = [col for col in self.columns_to_interpolate_ if col in available_columns]

        # Perform interpolation
        if 'Place_ID' in X_.columns:
            X_.set_index('Place_ID', inplace=True)
            X_[columns_to_interpolate] = X_.groupby(level='Place_ID')[columns_to_interpolate].transform(
                lambda group: group.interpolate(method='linear'))
            X_.reset_index(inplace=True)
        else:
            X_[columns_to_interpolate] = X_[columns_to_interpolate].interpolate(method='linear')

        # Filter columns_to_drop based on available columns
        columns_to_drop = [col for col in self.columns_to_drop if col in available_columns]

        # Drop specified columns that are present in X
        X_ = X_.drop(columns=columns_to_drop, errors='ignore')

        if self.columns_to_keep:
            X_ = X_[list(set(self.columns_to_keep) & set(available_columns))]

        return X_

    def get_feature_names_out(self, input_features=None):
        if self.columns_to_keep:
            return [col for col in self.columns_to_keep if col in self.train_columns]
        return [col for col in self.train_columns if col not in self.columns_to_drop]

##########################################################

def create_preprocessor(num_features=None, cat_features=None, bin_features=None, ord_features=None):
    transformers = []

    if num_features:
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])
        transformers.append(('num', num_transformer, num_features))

    if cat_features:
        nominal_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist', min_frequency=0.01))
        ])
        transformers.append(('nom', nominal_cat_transformer, cat_features))

    if bin_features:
        binary_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        transformers.append(('bin', binary_transformer, bin_features))

    if ord_features:
        ordinal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
        ])
        transformers.append(('ord', ordinal_transformer, ord_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

##########################################################

"""
class AdvancedTimeSeriesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='pmdarima', **kwargs):
        self.method = method
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.method == 'pmdarima':
            for col in X_.columns:
                if X_[col].isnull().sum() > 0:
                    X_[col] = impute_seasonal(X_[col], **self.kwargs)

        return X_
"""