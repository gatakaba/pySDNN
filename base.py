"""Base classes for all estimators."""
import numpy as np
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


class BaseEstimator(object):
    """Base class for all estimators"""

    @staticmethod
    def add_columns(X):
        X = check_array(X)
        intercepted_X = np.c_[np.ones(len(X)), X]
        return intercepted_X

    def score(self, X, y):
        check_is_fitted(self, ["X_train_", "y_train_"])
        X, y = check_X_y(X, y, multi_output=False)
        e = np.abs(self.predict(X) - y)
        return np.mean(e)
