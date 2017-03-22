#! /usr/bin/env python
# -*- coding:utf-8 -*-
""" Copyright (C) 2017 Yu Kabasawa

This is licensed under an MIT license. See the readme.md file
for more information.

"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted
from pysdnn import utils
from pysdnn.coding import PatternCoding


class PP(BaseEstimator):
    """ PPは複数のパーセプトロンを並列に並べ,それらの出力値の総計に応じて最終的な出力決定する教師あり学習モデルである.

    PPは3層のMLPにおいて,中間層の活性化関数をヘビサイド関数にし,
    中間層から出力層の結合荷重を固定したものとみなすことができる.

    Parameters
    ----------
    hidden_layer_num : int
        中間層の素子数
    eta: float
        学習係数
    verbose : bool
        詳細な出力を有効にする
    """

    def __init__(self, hidden_layer_num=300, eta=10 ** -3, verbose=False):
        self.W = None
        self.n_samples = None
        self.n_features = None
        self.eta = eta
        self.hidden_layer_num = hidden_layer_num
        self.verbose = verbose
        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2

    def activate_function(self, x):
        y = self.a * x + self.b
        return y

    def inverse_activate_function(self, y):
        x = (y - self.b) / self.a
        return x

    @staticmethod
    def _search_index(a, n_target, n_predict):
        # 修正するパーセプトロンを選ぶ
        error_num = int(round(np.abs(n_target - n_predict)))

        if error_num == 0:
            return []
        elif n_target > n_predict:
            """
                n_target > n_predictの場合、(n_target-n_predict)個のパーセプトロンを1が出るように修正
                修正するパーセプトロンは0以下のパーセプトロンの内最も内部電位が高いパーセプトロン
            """
            negative_perceptron_values = np.sort(a[a < 0])[::-1]
            if len(negative_perceptron_values) > error_num:
                fix_perceptron_values = negative_perceptron_values[:error_num]
            else:
                fix_perceptron_values = negative_perceptron_values
        else:
            positive_perceptron_values = np.sort(a[a > 0])
            if len(positive_perceptron_values) > error_num:
                fix_perceptron_values = positive_perceptron_values[:error_num]
            else:
                fix_perceptron_values = positive_perceptron_values

        index_list = []
        for fix_perceptron_value in fix_perceptron_values:
            index = np.where(a == fix_perceptron_value)[0][0]
            index_list.append(index)
        index_list = np.sort(index_list)
        return index_list

    def fit(self, X, y, is_intercepted=True):
        X, y = check_X_y(X, y, multi_output=False)
        n_samples, n_features = X.shape
        if is_intercepted:
            intercepted_X = utils.add_columns(X)
            self.W = np.random.normal(0, 1, size=[self.hidden_layer_num, n_features + 1])
        else:
            intercepted_X = X
            self.W = np.random.normal(0, 1, size=[self.hidden_layer_num, n_features])

        self.X_train_, self.y_train_ = np.copy(X), np.copy(y)

        for j in range(100):
            for i in (range(n_samples)):
                # 順伝播
                a = np.dot(self.W, intercepted_X[i])
                z = utils.step(a)
                n_predict = np.sum(z)
                n_target = self.inverse_activate_function(y[i])

                # 修正するパーセプトロンを選択
                index_list = self._search_index(a, n_target, n_predict)

                if not len(index_list) == 0:
                    self.W[index_list, :] += self.eta * np.sign(n_target - n_predict) * intercepted_X[i]

            if self.verbose:
                print(j, self.score(self.X_train_, self.y_train_))
        return self

    def predict(self, X, is_intercepted=True):
        check_is_fitted(self, ["X_train_", "y_train_"])
        prediction_list = []

        if is_intercepted:
            intercepted_X = utils.add_columns(X)
        else:
            intercepted_X = X
        for intercepted_x in intercepted_X:
            a = np.dot(self.W, intercepted_x)
            z = utils.step(a)
            a2 = np.sum(z)

            prediction = self.activate_function(a2)
            prediction_list.append(prediction)
        y = np.ravel(prediction_list)
        return y

    def score(self):
        pass


class PP_P(PP):
    def __init__(self, hidden_layer_num=300, eta=10 ** -3, verbose=False):
        super().__init__(hidden_layer_num, eta, verbose)

        self.W = None
        self.n_samples = None
        self.n_features = None
        self.eta = eta
        self.hidden_layer_num = hidden_layer_num
        self.verbose = verbose
        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2

        self.pc = None

    def fit(self, X, y):
        self.pc = PatternCoding(code_pattern_dim=100, input_division_num=100, reversal_num=1, input_dim=X.shape[1])
        code_X = self.pc.coding(X, 0, 1)
        super().fit(code_X, y, is_intercepted=False)

    def predict(self, X):
        code_X = self.pc.coding(X, 0, 1)
        y = super().predict(code_X, is_intercepted=False)
        return y
