#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import unittest

from pysdnn import SDNN
from sklearn.utils.estimator_checks import check_estimator


class TestSDNN(unittest.TestCase):
    def setUp(self):
        # make sample data
        n_train_samples = 200
        n_test_samples = 100
        self.train_X = np.random.uniform(0, 1, size=[n_train_samples, 2])
        self.train_y = np.random.normal(size=n_train_samples)

        self.test_X = np.random.uniform(0, 1, size=[n_test_samples, 2])
        self.test_y = np.random.normal(size=n_train_samples)

        self.sdnn = SDNN()

    def test_fit(self):
        self.sdnn.fit(self.train_X, self.train_y)

    def test_predict(self):
        self.sdnn.fit(self.train_X, self.train_y)
        self.sdnn.predict(self.test_X)

    """
    def test_estimator(self):
        check_estimator(MultiLayerPerceptronRegression)

    """
