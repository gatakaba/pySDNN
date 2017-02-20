# coding:utf-8

import numpy as np


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
