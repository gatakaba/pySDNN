# coding:utf-8
import unittest
import numpy as np
from pysdnn.coding import PatternCoding
from pysdnn.coding import SelectiveDesensitization


class Test_Coding(unittest.TestCase):
    def setUp(self):
        self.binary_vector_dim = 8
        self.division_num = 6
        self.reversal_num = 2

        self.pc = PatternCoding(self.binary_vector_dim, self.division_num, self.reversal_num)
        self.feature_num = 5
        self.pc.make_binary_vector_table(self.feature_num)

    def test_array_size(self):
        assert self.pc._make_binary_vector_table_1d().shape == (self.division_num, self.binary_vector_dim)
        assert self.pc.binary_vector_table.shape == (self.feature_num, self.division_num, self.binary_vector_dim)

    def test_num_to_pattern(self):
        # check 1d
        x = np.random.random(size=self.feature_num)
        # check size
        assert self.pc.num_to_pattern(x).shape == (self.binary_vector_dim * self.feature_num,)
