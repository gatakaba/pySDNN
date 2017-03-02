# coding:utf-8
import unittest
from pysdnn.coding import PatternCoding
from pysdnn.coding import SelectiveDesensitization


class Test_Coding(unittest.TestCase):
    def setUp(self):
        self.binary_vector_dim = 100
        self.division_num = 100
        self.reversal_num = 1

        self.pc = PatternCoding(self.binary_vector_dim, self.division_num, self.reversal_num)
        self.feature_num = 10
        self.pc.make_binary_vector_tables(self.feature_num)

    def test_array_size(self):
        assert self.pc._make_binary_vector_table().shape == (self.division_num, self.binary_vector_dim)
