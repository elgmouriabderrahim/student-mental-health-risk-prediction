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
