#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
PPは複数のパーセプトロンを並列に並べ,それらの出力値の総計に応じて最終的な出力決定する教師あり学習モデルである.

PPは3層のMLPにおいて,中間層の活性化関数をヘビサイド関数にし,中間層から出力層の結合荷重を固定したものとみなすことができる.
"""

from pysdnn.base_network import BaseNetwork
from pysdnn.coding import PatternCoding
from pysdnn.utils import add_interception


class PP_A(BaseNetwork):
    """Parallel Peceptron Analogue

    Parameters
    ----------
    hidden_layer_num : int, optional (default = 280)
        中間素子数
    verbose : bool, optional (default = False)
        詳細な出力を有効化
    """

    def __init__(self, hidden_layer_num=300, verbose=False):
        super().__init__(hidden_layer_num, verbose)

        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2

    def fit(self, X, y, learning_num=100, eta=10 ** -3):
        """Fit the SDNN model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (sample_num, input_dim)
            Training vectors.
        y : array-like, shape = (sample_num,)
            Target values.
        learning_num : int, optional (default = 100)
            学習回数
        eta : float, optional (default = 0.001)
            学習率

        Returns
        -------
        self : object
            Returns self.
        """
        intercepted_X = add_interception(X)
        super().fit(intercepted_X, y, learning_num, eta)

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
        y = super().predict(intercepted_X)
        return y


class PP_P(BaseNetwork):
    """ Parallel Peceptron Pattern

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
    verbose : bool, optional (default = False)
        詳細な出力を有効化
    """

    def __init__(self, code_pattern_dim=100, input_division_num=100, reversal_num=1, hidden_layer_num=300,
                 verbose=False):
        super().__init__(hidden_layer_num, verbose)
        self.code_pattern_dim = code_pattern_dim
        self.input_division_num = input_division_num
        self.reversal_num = reversal_num

        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2

        self.pc = None

    def fit(self, X, y, learning_num=100, eta=10 ** -3):
        """Fit the SDNN model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (sample_num, input_dim)
            Training vectors.
        y : array-like, shape = (sample_num,)
            Target values.
        learning_num : int, optional (default = 100)
            学習回数
        eta : float, optional (default = 0.001)
            学習率

        Returns
        -------
        self : object
            Returns self.
        """
        intercepted_X = add_interception(X)
        self.pc = PatternCoding(self.code_pattern_dim, self.input_division_num, self.reversal_num,
                                input_dim=intercepted_X.shape[1])
        code_X = self.pc.coding(intercepted_X, 0, 1)
        super().fit(code_X, y, learning_num, eta)

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
        code_X = self.pc.coding(intercepted_X, 0, 1)
        y = super().predict(code_X)
        return y
