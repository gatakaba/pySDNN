#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""docstring"""
from pysdnn.coding import PatternCoding
import numpy as np

code_pattern_dim = 8
division_num = 6
reversal_num = 2
input_dim = 5

x = np.random.random(size=input_dim)
print(x)

pc = PatternCoding(code_pattern_dim, division_num, reversal_num, input_dim)

print(pc.coding(x))
print(pc.coding(x).shape)
