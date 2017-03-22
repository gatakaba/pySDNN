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


def linear(X):
    """ 線形関数

    .. math::
        f(x) = \\sum_{i} x_{i}

    Parameters
    ----------
    X  : array-like, shape = (samples_num, input_dim)
        入力データ

    Returns
    -------
    y : array-like, shape = (samples_num,)
        計算結果
    """
    return np.sum(X, axis=1)


def nonaka(X):
    """ 野中関数[1]_

    .. math::
        f(x,y) =
        \\begin{cases}
            1 ((x-0.5)^{2}+(y-0.5)^{2} < 0.04) \\\\
            \\frac{1+x}{2}sin^{2}(6 \pi \sqrt{xy^{2}}) (otherwise)
        \\end{cases}

    .. [1] 野中和明, 田中文英, and 森田昌彦. "階層型ニューラルネットの 2 変数関数近似能力の比較." 電子情報通信学会論文誌 D 94.12 (2011): 2114-2125.

    Parameters
    ----------
    X  : array-like, shape = (samples_num, 2)
        入力データ


    Returns
    -------
    y : array-like, shape = (samples_num,)
        計算結果
    """

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


def cylinder(X):
    """ 円筒関数

    .. math::
        f(x,y) =
        \\begin{cases}
            1 ((x-0.5)^{2}+(y-0.5)^{2} < 0.04) \\\\
            0 (otherwise)
        \\end{cases}

    Parameters
    ----------
    X  : array-like, shape = (samples_num, 2)
        入力データ

    Returns
    -------
    y : array-like, shape = (samples_num,)
        計算結果
    """
    t = []
    for x in X:
        if (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < 0.4 ** 2:
            t.append(1.0)
        else:
            t.append(0.0)
    return np.array(t)


def gauss(X):
    """ ガウス関数

    .. math::
        f(x,y) = \\exp(-(x-0.5)^{2} -(y-0.5)^{2})

    Parameters
    ----------
    X  : array-like, shape = (samples_num, 2)
        入力データ

    Returns
    -------
    y : array-like, shape = (samples_num,)
        計算結果
    """

    return np.exp(-(X[:, 0] - 0.5) ** 2 - (X[:, 1] - 0.5) ** 2)


