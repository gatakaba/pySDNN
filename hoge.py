#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""docstring"""
from pysdnn.coding import PatternCoding
import numpy as np

code_pattern_dim = 8
division_num = 10
reversal_num = 1

input_dim = 2

x = np.random.random(size=input_dim)

pc = PatternCoding(code_pattern_dim, division_num, reversal_num, input_dim)

x = np.array([0.5, 0.5])
y = np.array([0.6, 0.6])

print(pc.coding(x) - pc.coding(y))

print(pc.coding(x).shape)
