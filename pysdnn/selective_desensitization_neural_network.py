#! /usr/bin/env python
# -*- coding:utf-8 -*-

from pysdnn.base_network import BaseNetwork
from pysdnn.coding import SelectiveDesensitization
from pysdnn.utils import add_interception


class SDNN(BaseNetwork):
    """Selective Desensitization Neural Network

    Parameters
    ----------
    code_pattern_dim : int, optional (default = 100)
        パターンコードベクトルの次元数 n
    input_division_num : int, optional (default = 100)
        実数の分割数 q
    reversal_num : int, optinal (default = 1)
        反転数 r
    hidden_layer_num : int, optional (default = 280)
        中間素子数
    eta : float, optional (default = 0.001)
        学習率
    verbose : bool, optional (default = False)
        詳細な出力を有効化
    """

    def __init__(self, code_pattern_dim=100, input_division_num=100, reversal_num=1, hidden_layer_num=200, eta=10 ** -3,
                 verbose=False):
        super().__init__(hidden_layer_num, eta, verbose)

        self.code_pattern_dim = code_pattern_dim
        self.input_division_num = input_division_num
        self.reversal_num = reversal_num
        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2
        self.sd = None

    def fit(self, X, y):
        """Fit the SDNN model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (sample_num, input_dim)
            Training vectors.
        y : array-like, shape = (sample_num,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        intercepted_X = add_interception(X)
        input_dim = intercepted_X.shape[1]
        self.sd = SelectiveDesensitization(self.code_pattern_dim, self.input_division_num, self.reversal_num,
                                           input_dim=input_dim)
        sd_code_X = self.sd.coding(intercepted_X, 0, 1)
        super().fit(sd_code_X, y)
        return self

    def predict(self, X):
        """Perform regression on samples in X.

        Parameters
        ----------
        X : array-like, shape = (sample_num,input_dim)

        Returns
        -------
        y_pred : array-like, shape = (sample_num, )
        """
        intercepted_X = add_interception(X)
        sd_code_X = self.sd.coding(intercepted_X, 0, 1)
        y_pred = super().predict(sd_code_X)
        return y_pred
