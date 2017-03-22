#! /usr/bin/env python
# -*- coding:utf-8 -*-
""" Copyright (C) 2017 Yu Kabasawa

This is licensed under an MIT license. See the readme.md file
for more information.
"""

import numpy as np
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


def add_columns(input_array):
    """ 切片を加える

    切片は末尾に加えられる

    .. math::

        \mathbf{p}_{i,j} = \\frac{\mathbf{p}_{i} + 1}{2} \mathbf{p}_{j}

        \mathbf{p}_{j,i} = \\frac{\mathbf{p}_{j} + 1}{2} \mathbf{p}_{i}


    Parameters
    ----------
     input_array : array-like, shape = (samples_num, input_dim)
            入力データ

    Returns
    -------
    intercepted_X : array of shape = (samples_num, input_dim + 1)
            切片を加えられた入力データ
    """

    intercepted_array = np.c_[input_array, np.ones(len(input_array))]
    return intercepted_array


def linear(X):
    return np.sum(X, axis=1) / 2.0


def nonaka(X):
    r = 0.2
    if len(X.shape) == 1:
        if (X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 < r ** 2:
            return 1
        else:
            return (1 + X[0]) / 2.0 * np.sin(6 * np.pi * X[0] ** 0.5 * X[1] ** 2) ** 2
    t = []
    for x in X:
        if (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < r ** 2:
            t.append(1)
        else:
            t.append(
                (1 + x[0]) / 2.0 * np.sin(6 * np.pi * x[0] ** 0.5 * x[1] ** 2) ** 2)
    t = np.array(t)

    return t


def wave(X):
    return np.cos(2 * np.pi * X[:, 0]) * np.sin(2 * np.pi * X[:, 1])


def cylinder(X):
    t = []
    for x in X:
        if (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < 0.4 ** 2:
            t.append(1.0)
        else:
            t.append(0.0)
    return np.array(t)


def gauss(X):
    return np.exp(-(X[:, 0] - 0.5) ** 2 - (X[:, 1] - 0.5) ** 2)


if __name__ == "__main__":
    pass
