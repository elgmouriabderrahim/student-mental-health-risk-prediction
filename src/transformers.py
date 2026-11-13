"""Custom scikit-learn transformers used by the prediction pipeline."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    """Remove columns that must not be used as model features."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns).copy()


class ScreenTimeConsistency(BaseEstimator, TransformerMixin):
    """Mark social-media use above total screen time as missing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        invalid = X["Social_Media_Hours"] > X["Screen_Time_Hours"]
        X.loc[invalid, "Social_Media_Hours"] = np.nan
        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Cap selected numeric columns with IQR bounds learned during fit."""

    def __init__(self, columns, multiplier=1.5):
        self.columns = columns
        self.multiplier = multiplier

    def fit(self, X, y=None):
        q1 = X[self.columns].quantile(0.25)
        q3 = X[self.columns].quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.multiplier * iqr
        self.upper_bounds_ = q3 + self.multiplier * iqr
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[:, self.columns] = X[self.columns].clip(
            self.lower_bounds_, self.upper_bounds_, axis=1
        )
        return X
