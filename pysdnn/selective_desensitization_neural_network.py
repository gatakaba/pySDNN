#! /usr/bin/env python
# -*- coding:utf-8 -*-

from pysdnn.base_network import BaseNetwork
from pysdnn.coding import SelectiveDesensitization


class SDNN(BaseNetwork):
    """Selective Desensitization Neural Network"""

    def __init__(self, hidden_layer_num=200, eta=10 ** -3, verbose=False):
        super().__init__(hidden_layer_num, eta, verbose)
        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2
        self.sd = None

    def fit(self, X, y):
        self.sd = SelectiveDesensitization(code_pattern_dim=100, input_division_num=100, reversal_num=1,
                                           input_dim=X.shape[1])
        sd_code_X = self.sd.coding(X, 0, 1)
        super().fit(sd_code_X, y)

    def predict(self, X):
        sd_code_X = self.sd.coding(X, 0, 1)
        y = super().predict(sd_code_X)
        return y
