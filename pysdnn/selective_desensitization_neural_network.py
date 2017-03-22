#! /usr/bin/env python
# -*- coding:utf-8 -*-

from pysdnn.base_network import BaseNetwork
from pysdnn.coding import SelectiveDesensitization
from pysdnn.utils import add_interception


class SDNN(BaseNetwork):
    """Selective Desensitization Neural Network"""

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
        """Fit the SDNN model according to the given training data."""

        intercepted_X = add_interception(X)
        self.sd = SelectiveDesensitization(self.code_pattern_dim, self.input_division_num, self.reversal_num,
                                           input_dim=intercepted_X.shape[1])
        sd_code_X = self.sd.coding(intercepted_X, 0, 1)
        super().fit(sd_code_X, y)

    def predict(self, X):
        intercepted_X = add_interception(X)
        sd_code_X = self.sd.coding(intercepted_X, 0, 1)
        y = super().predict(sd_code_X)
        return y
