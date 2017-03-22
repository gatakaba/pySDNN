#! /usr/bin/env python
# -*- coding:utf-8 -*-
""" Copyright (C) 2017 Yu Kabasawa

This is licensed under an MIT license. See the readme.md file
for more information.

"""

from pysdnn.base_network import BaseNetwork
from pysdnn.coding import PatternCoding
from pysdnn import utils


class PP_A(BaseNetwork):
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
