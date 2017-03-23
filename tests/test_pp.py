import numpy as np
import unittest

from pysdnn.parallel_perceptron import PP_A, PP_P


class TestPP_A(unittest.TestCase):
    def setUp(self):
        # make sample data
        n_train_samples = 20
        n_test_samples = 10
        self.train_X = np.random.uniform(0, 1, size=[n_train_samples, 3])
        self.train_y = np.random.normal(size=n_train_samples)

        self.test_X = np.random.uniform(0, 1, size=[n_test_samples, 3])
        self.test_y = np.random.normal(size=n_train_samples)

    def test_pattern(self):
        pass

    def test_fit(self):
        self.pp = PP_A()
        self.pp.fit(self.train_X, self.train_y)

    def test_predict(self):
        self.pp = PP_A()
        self.pp.fit(self.train_X, self.train_y)
        self.pp.predict(self.test_X)


class TestPP_P(unittest.TestCase):
    def setUp(self):
        # make sample data
        n_train_samples = 20
        n_test_samples = 10
        self.train_X = np.random.uniform(0, 1, size=[n_train_samples, 3])
        self.train_y = np.random.normal(size=n_train_samples)

        self.test_X = np.random.uniform(0, 1, size=[n_test_samples, 3])
        self.test_y = np.random.normal(size=n_train_samples)

    def test_fit(self):
        self.pp_p = PP_P()
        self.pp_p.fit(self.train_X, self.train_y)

    def test_predict(self):
        self.pp_p = PP_P()
        self.pp_p.fit(self.train_X, self.train_y)
        self.pp_p.predict(self.test_X)
