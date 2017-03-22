#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
PPは複数のパーセプトロンを並列に並べ,それらの出力値の総計に応じて最終的な出力決定する教師あり学習モデルである.

PPは3層のMLPにおいて,中間層の活性化関数をヘビサイド関数にし,中間層から出力層の結合荷重を固定したものとみなすことができる.
"""

from pysdnn.base_network import BaseNetwork
from pysdnn.coding import PatternCoding
from pysdnn import utils


class PP_A(BaseNetwork):
    """ Parallel Peceptron Analogue"""
    def __init__(self, hidden_layer_num=300, eta=10 ** -3, verbose=False):
        super().__init__(hidden_layer_num, eta, verbose)

        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2

    def fit(self, X, y):
        intercepted_X = utils.add_columns(X)
        super().fit(intercepted_X, y)

    def predict(self, X):
        intercepted_X = utils.add_columns(X)
        y = super().predict(intercepted_X)
        return y


class PP_P(BaseNetwork):
    """ Parallel Peceptron Pattern"""
    def __init__(self, hidden_layer_num=300, eta=10 ** -3, verbose=False):
        super().__init__(hidden_layer_num, eta, verbose)

        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2

        self.pc = None

    def fit(self, X, y):
        self.pc = PatternCoding(code_pattern_dim=100, input_division_num=100, reversal_num=1, input_dim=X.shape[1])
        code_X = self.pc.coding(X, 0, 1)
        super().fit(code_X, y)

    def predict(self, X):
        code_X = self.pc.coding(X, 0, 1)
        y = super().predict(code_X)
        return y
