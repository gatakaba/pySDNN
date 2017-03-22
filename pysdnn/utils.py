#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np


def add_interception(input_array):
    """ 切片を加える

    各データの末尾に切片1を加える

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
