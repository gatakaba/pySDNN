#! /usr/bin/env python
# -*- coding:utf-8 -*-


import unittest
import numpy as np
from pysdnn import utils


class TestCoding(unittest.TestCase):
    def setUp(self):
        pass

    def test_add_interception(self):
        sample_num = 100
        input_dim = 3
        input_array = np.random.random(size=[sample_num, input_dim])
        y = utils.add_interception(input_array)
        assert y.shape == (sample_num, input_dim + 1)
        assert y[sample_num - 5, -1] == 1
