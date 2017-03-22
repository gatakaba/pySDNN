#! /usr/bin/env python
# -*- coding:utf-8 -*-
""" Copyright (C) 2017 Yu Kabasawa

This is licensed under an MIT license. See the readme.md file
for more information.
"""

import numpy as np


def add_columns(input_array):
    """ 切片を加える

    1を各データの末尾に加える

    .. math::

        X = \\begin{pmatrix}
                        a & b \\\\
                        c & d \\\\
                        e & f
                        \\end{pmatrix}

        inpteceptedX = \\begin{pmatrix}
                        a & b & 1 \\\\
                        c & d & 1 \\\\
                        e & f & 1
                        \\end{pmatrix}

    Parameters
    ----------
     input_array : array-like, shape = (samples_num, input_dim)
            入力データ

    Returns
    -------
    intercepted_array : array of shape = (samples_num, input_dim + 1)
            切片を加えられた入力データ
    """

    intercepted_array = np.c_[input_array, np.ones(len(input_array))]
    return intercepted_array


def step(x):
    """ step function

    .. math::
        f(x) =
        \\begin{cases}
            1 (x \geq 0) \\\\
            0 (x < 0)
        \\end{cases}

    Parameters
    ----------
    x : float or array-like, shape = (sample_num,)
        入力データ

    Returns
    -------
    y : float or array-like, shape = (sample_num,)
        計算結果
    """
    y = (np.sign(x) + 1) / 2.0
    return y

