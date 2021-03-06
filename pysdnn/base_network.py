#! /usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted


def _step(x):
    """ ステップ関数

    .. math::
        f(x) =
        \\begin{cases}
            1 (x > 0) \\\\
            0.5 (x = 0) \\\\
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


def _scaling_function(x, a, b):
    """ スケーリング関数

    .. math::
        f(x) =  a x + b

    Parameters
    ----------
    x : float or array-like, shape = (sample_num,)
        入力データ
    a : float
        傾き
    b : float
        切片

    Returns
    -------
    y : float or array-like, shape = (sample_num,)
        計算結果
    """
    y = a * x + b
    return y


def _inverse__scaling_function(y, a, b):
    """ スケーリング関数の逆関数

    .. math::
        x =  \\frac{y-b}{a}

    Parameters
    ----------
    y : float or array-like, shape = (sample_num,)
        入力データ
    a : float
        傾き
    b : float
        切片

    Returns
    -------
    x : float or array-like, shape = (sample_num,)
        計算結果
    """

    x = (y - b) / a
    return x


class BaseNetwork(BaseEstimator):
    """ Base NetworkクラスはPP,SDNNの抽象クラスです.

    本クラスはSDNN[R1]の第3-5層目の順伝播及び教師あり学習を用いた第3-4層の重み荷重調節機能を有します.


    .. [1] 野中和明, 田中文英, and 森田昌彦. "階層型ニューラルネットの 2 変数関数近似能力の比較." 電子情報通信学会論文誌 D 94.12 (2011): 2114-2125.

    Parameters
    ----------
    hidden_layer_num : int
        中間層の素子数
    verbose : bool
        詳細な出力を有効にする
    """

    def __init__(self, hidden_layer_num=300, verbose=False):
        self.hidden_layer_num = hidden_layer_num
        self.verbose = verbose

        self.eta = None
        self.W = None
        self.n_samples = None
        self.n_features = None
        self.a = 1.4 / self.hidden_layer_num
        self.b = -0.2

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

    def fit(self, X, y, learning_num, eta):
        """Fit the BaseNetwork model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (sample_num, input_dim)
            Training vectors.
        y : array-like, shape = (sample_num,)
            Target values.
        learning_num : int
            学習回数
        eta : float
            学習率

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, multi_output=False)
        n_samples, n_features = X.shape
        self.eta = eta
        self.W = np.random.normal(0, 1, size=[self.hidden_layer_num, n_features])
        self.X_train_, self.y_train_ = np.copy(X), np.copy(y)

        for j in range(learning_num):
            for i in (range(n_samples)):
                # 順伝播
                a = np.dot(self.W, X[i])
                z = _step(a)
                n_predict = np.sum(z)
                n_target = _inverse__scaling_function(y[i], self.a, self.b)
                # 修正するパーセプトロンを選択
                index_list = self._search_index(a, n_target, n_predict)
                # パーセプトロンを修正
                if not len(index_list) == 0:
                    self.W[index_list, :] += self.eta * np.sign(n_target - n_predict) * X[i]

            if self.verbose:
                print(j, self.score(self.X_train_, self.y_train_))
        return self

    def predict(self, X):
        """Perform regression on samples in X.

        Parameters
        ----------
        X : array-like, shape = (sample_num,input_dim)

        Returns
        -------
        y_pred : array-like, shape = (sample_num, )
        """
        check_is_fitted(self, ["X_train_", "y_train_"])
        prediction_list = []

        for x in X:
            # 順伝播
            a = np.dot(self.W, x)
            z = _step(a)
            a2 = np.sum(z)
            prediction = _scaling_function(a2, self.a, self.b)
            prediction_list.append(prediction)
        y = np.ravel(prediction_list)
        return y

    def score(self):
        pass
