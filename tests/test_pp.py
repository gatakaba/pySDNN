import numpy as np
import unittest

from pysdnn import PP
from sklearn.utils.estimator_checks import check_estimator


class Test_PP(unittest.TestCase):
    def setUp(self):
        # make sample data
        n_train_samples = 200
        n_test_samples = 100
        self.train_X = np.random.uniform(0, 1, size=[n_train_samples, 3])
        self.train_y = np.random.normal(size=n_train_samples)

        self.test_X = np.random.uniform(0, 1, size=[n_test_samples, 3])
        self.test_y = np.random.normal(size=n_train_samples)

        self.pp = PP()

    def test_fit(self):
        self.pp.fit(self.train_X, self.train_y)

    def test_predict(self):
        self.pp.fit(self.train_X, self.train_y)
        self.pp.predict(self.test_X)

    """
    def test_estimator(self):
        check_estimator(MultiLayerPerceptronRegression)

    """