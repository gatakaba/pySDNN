#! /usr/bin/env python
# -*- coding:utf-8 -*-


import unittest
import numpy as np
from pysdnn import utils


class TestCoding(unittest.TestCase):
    def setUp(self):
        pass

    def test_add_columns(self):
        sample_num = 100
        input_dim = 3

        input_array = np.random.random(size=[sample_num, input_dim])

        y = utils.add_columns(input_array)

        assert y.shape == (sample_num, input_dim + 1)
        assert y[sample_num - 5, -1] == 1

    def test_step(self):
        assert utils.step(-3) == 0.0
        assert utils.step(0) == 0.5
        assert utils.step(7) == 1.0
