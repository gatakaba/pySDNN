#! /usr/bin/env python
# -*- coding:utf-8 -*-

import unittest
import numpy as np
from pysdnn.coding import PatternCoding
from pysdnn.coding import SelectiveDesensitization


class TestCoding(unittest.TestCase):
    def setUp(self):
        self.code_pattern_dim = 8
        self.division_num = 6
        self.reversal_num = 2
        self.input_dim = 5

        self.pc = PatternCoding(self.code_pattern_dim, self.division_num, self.reversal_num, self.input_dim)

    def test_code_pattern_size(self):
        assert self.pc._create_code_pattern().shape == (self.division_num, self.code_pattern_dim)
        assert self.pc.code_pattern_table.shape == (self.input_dim, self.division_num, self.code_pattern_dim)

    def test_coding_1d(self):
        x = np.random.random(size=self.input_dim)
        assert self.pc.coding(x).shape == (self.code_pattern_dim * self.input_dim,)

    def test_coding_2d(self):
        n_samples = 10
        x = np.random.random(size=[n_samples, self.input_dim])
        assert self.pc.coding(x).shape == (n_samples, self.code_pattern_dim * self.input_dim)

    def test_coding_input_range(self):
        x = np.random.random(size=[8, self.input_dim])

        self.pc.coding(x)
