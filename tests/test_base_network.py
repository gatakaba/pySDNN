#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import unittest

from pysdnn import BaseNetwork


class TestBaseNetwork(unittest.TestCase):
    def setUp(self):
        # make sample data
        n_train_samples = 20
        n_test_samples = 10
        self.train_X = np.random.uniform(0, 1, size=[n_train_samples, 3])
        self.train_y = np.random.normal(size=n_train_samples)

        self.test_X = np.random.uniform(0, 1, size=[n_test_samples, 3])
        self.test_y = np.random.normal(size=n_train_samples)

        self.bn = BaseNetwork()

    def test_fit(self):
        self.bn.fit(self.train_X, self.train_y, 10, 0.001)

    def test_predict(self):
        self.bn.fit(self.train_X, self.train_y, 10, 0.001)
        self.bn.predict(self.test_X)
