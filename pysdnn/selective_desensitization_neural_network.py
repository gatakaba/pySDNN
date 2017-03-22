# coding:utf-8
""" Copyright (C) 2017 Yu Kabasawa

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import numpy as np

from sklearn.base import BaseEstimator
from pysdnn.coding import SelectiveDesensitization
from pysdnn.parallel_perceptron import PP


class SDNN(PP):
    def __init__(self, hidden_layer_num=200, eta=10 ** -3, verbose=False):
        super().__init__(hidden_layer_num, eta, verbose)

        self.W = None
        self.n_samples = None
        self.n_features = None
        self.eta = eta
        self.hidden_layer_num = hidden_layer_num
        self.verbose = verbose
        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2

        self.pc = None

    def fit(self, X, y):
        self.pc = SelectiveDesensitization(code_pattern_dim=100, input_division_num=100, reversal_num=1,
                                           input_dim=X.shape[1])
        code_X = self.pc.coding(X, 0, 1)
        super().fit(code_X, y, is_intercepted=False)

    def predict(self, X):
        code_X = self.pc.coding(X, 0, 1)
        y = super().predict(code_X, is_intercepted=False)
        return y
