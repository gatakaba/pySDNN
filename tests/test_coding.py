#! /usr/bin/env python
# -*- coding:utf-8 -*-

import unittest
import numpy as np
from pysdnn.coding import PatternCoding
from pysdnn.coding import SelectiveDesensitization


class TestCoding(unittest.TestCase):
    def setUp(self):
        self.code_pattern_dim = 13
        self.division_num = 17
        self.reversal_num = 2
        self.input_dim = 5

        self.pc = PatternCoding(self.code_pattern_dim, self.division_num, self.reversal_num, self.input_dim)

    def test_pattern_size(self):
        assert self.pc._create_code_pattern().shape == (self.division_num, self.code_pattern_dim)
        assert self.pc.code_pattern_table.shape == (self.input_dim, self.division_num, self.code_pattern_dim)

    def test_coding_scalar_input(self):
        pc = PatternCoding(self.code_pattern_dim, self.division_num, self.reversal_num, 1)
        x = np.array(np.random.random())
        assert pc.coding(x).shape == (self.code_pattern_dim,)

    def test_coding_1d(self):
        x = np.random.random(size=self.input_dim)
        assert self.pc.coding(x).shape == (self.code_pattern_dim * self.input_dim,)

    def test_coding_2d(self):
        n_samples = 10
        x = np.random.random(size=[n_samples, self.input_dim])
        assert self.pc.coding(x).shape == (n_samples, self.code_pattern_dim * self.input_dim)

    def test_input_range(self):
        x = np.random.normal(size=[100, self.input_dim]) * 100
        self.pc.coding(x)
        self.pc.coding(x, -10, 10)
        self.pc.coding(x, np.min(x, axis=0), np.max(x, axis=0))


class TestSelectiveDesensitization(unittest.TestCase):
    def setUp(self):
        self.code_pattern_dim = 13
        self.division_num = 17
        self.reversal_num = 2
        self.input_dim = 5

        self.sd = SelectiveDesensitization(self.code_pattern_dim, self.division_num, self.reversal_num, self.input_dim)

    def test_coding_1d(self):
        x = np.random.random(size=self.input_dim)
        assert self.sd.coding(x).shape == (self.input_dim * (self.input_dim - 1) * self.code_pattern_dim,)

    def test_coding_2d(self):
        n_samples = 10
        x = np.random.random(size=[n_samples, self.input_dim])
        assert self.sd.coding(x).shape == (n_samples, self.input_dim * (self.input_dim - 1) * self.code_pattern_dim)

    def test_input_range(self):
        x = np.random.normal(size=[100, self.input_dim]) * 100
        self.sd.coding(x)
        self.sd.coding(x, -10, 10)
        self.sd.coding(x, np.min(x, axis=0), np.max(x, axis=0))
