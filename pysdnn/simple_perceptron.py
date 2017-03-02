# coding:utf-8

"""
simple perceptron
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from pysdnn.utilities import add_columns


class SP(BaseEstimator, RegressorMixin):
    """
    simple perceptron model
        f(x) = sign(<w, x>)

    Attributes
    ----------
    w : array, shape = (n_features,)
        weight variable
    """

    def __init__(self, eta=10 ** -3, verbose=False):
        """
        Parameters
        ----------
        eta : float
            learning rate

             when eta = 1 , eta is named
             eta=1の時固定増分誤り訂正法と呼ぶ
        """
        self.w = None
        self.finish_value = 10 ** -5
        self.eta = eta
        self.verbose = True

    def fit(self, X, y):

        """Fit linear model with perceptron criterion.

            E_{p} = - \Sigma_{n} \mathbf{w}^{y} \mathbf{\phi(x_n)} y_{n}

            w_{\tau+1} = w_{\tau} − \tau \nabla Ep(w) = w{\tau} + \tau \phi(xn) y_{n}


        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Train samples.

        y : array-like, shape = (n_samples)
            Target values for X.

        Returns
        -------
        self : returns an instance of self.
        """

        X, y = check_X_y(X, y, multi_output=False)
        self.X_train_, self.y_train_ = np.copy(X), np.copy(y)
        n_samples, n_features = X.shape
        intercepted_X = add_columns(X)
        self.w = np.random.normal(0, 1, size=n_features + 1)

        for i in range(1000):
            for i in range(n_samples):
                self.w += self.eta * intercepted_X[i] * y[i]
            if self.verbose:
                print(self.score(self.X_train_, self.y_train_))
        return self

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        check_is_fitted(self, ["X_train_", "y_train_"])
        intercepted_X = add_columns(X)
        y = np.sign(np.dot(intercepted_X, self.w))
        return y
