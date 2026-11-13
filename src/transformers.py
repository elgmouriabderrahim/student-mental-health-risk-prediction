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
