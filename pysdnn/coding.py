#! /usr/bin/env python
# -*- coding:utf-8 -*-
""" Copyright (C) 2017 Yu Kabasawa

This is licensed under an MIT license. See the readme.md file
for more information.
"""

import numpy as np


class PatternCoding(object):
    """ PatternCodingクラスは実数とコードパターンの対応関係の管理を行うクラスです.

    コードパターンとは :math:`\{-1,1\}` を要素とする :math:`M` 次元ベクトルです.

    PatternCodingクラスでは :math:`N` 次元の実数ベクトルを一括して扱い,
    コーディングの出力結果は :math:`N \\ast M` 次元ベクトルです.

    Parameters
    ----------
    code_pattern_dim : int
        コードパターンベクトルの次元数 n
    input_division_num : int
        実数の分割数 q
    reversal_num : int
        反転数 r
    input_dim : int
        入力データの次元数 N
    """

    def __init__(self, code_pattern_dim, input_division_num, reversal_num, input_dim):
        self.code_pattern_dim = code_pattern_dim
        self.input_division_num = input_division_num
        self.reversal_num = reversal_num
        self.input_dim = input_dim

        # コードパターン対応表を作成
        self.code_pattern_table = self._create_code_pattern_table()

    def _create_code_pattern_table(self):
        """ 実数とコードパターンの対応表を作成する
        Returns
        -------
        code_pattern_table : ndarray, shape = (input_dim, division_num, code_pattern_dim)
            コードパターン対応表
            code_pattern_table[i,j]はi次元目のj番目のパターンを表す
        """

        code_pattern_table = np.empty([self.input_dim, self.input_division_num, self.code_pattern_dim])
        for i in range(self.input_dim):
            code_pattern_table[i] = self._create_code_pattern()

        return code_pattern_table

    def _create_code_pattern(self):
        """ コードパターンを作成する

        コードパターンは以下の条件を満たす
        - 入力次元ごとに異なるパターンを持つ
        - 重複が無い
        - -1と1の数の数が等しい

        Returns
        -------
        code_pattern : ndarray, shape = (division_num,code_pattern_dim)
            コードパターン
            code_pattern[i]はi番目のパターンを表す
        """

        code_pattern = []
        binary_vector = np.ones(self.code_pattern_dim)
        binary_vector[:int(self.code_pattern_dim / 2)] = -1
        np.random.shuffle(binary_vector)
        code_pattern.append(binary_vector)

        while len(code_pattern) < self.input_division_num:
            while True:
                tmp_binary_vector = np.copy(code_pattern[-1])
                # select reverse index
                index1 = list(
                    np.random.choice(np.where(tmp_binary_vector == -1)[0], size=self.reversal_num, replace=False))
                index2 = list(
                    np.random.choice(np.where(tmp_binary_vector == 1)[0], size=self.reversal_num, replace=False))
                index = index1 + index2

                # reverse selected index
                tmp_binary_vector[index] *= -1

                # if tmp_binary_vector is included in code_pattern, add to code_pattern
                if not any((tmp_binary_vector == x).all() for x in code_pattern):
                    code_pattern.append(tmp_binary_vector)
                    break

        return np.array(code_pattern)

    def _real_to_code(self, x, low, high):
        """ 実数xをコードパターンに変換する

        Parameters
        ----------
        x : ndarray, shape = (input_dim,)
            入力値
        low : int or float or ndarray, shape =(input_dim,), optional
            入力値の下限
        high : int or float or ndarray, shape =(input_dim,), optional
            入力値の上限

        Returns
        -------
            code_pattern : ndarray, shape = (input_dim * code_pattern_dim,)
        """

        if isinstance(low, float) or isinstance(low, int):
            low_array = np.ones(self.input_dim) * low
        else:
            low_array = low

        if isinstance(high, float) or isinstance(high, int):
            high_array = np.ones(self.input_dim) * high
        else:
            high_array = high

        code_list = []
        for i, element in enumerate(x):
            scaled_element = (element - low_array[i]) / (high_array[i] - low_array[i])

            index = int(np.floor(scaled_element * self.input_division_num))

            # 正規化
            if index < 0:
                index = 0
            if index > self.input_division_num - 1:
                index = self.input_division_num - 1

            code_list.append(self.code_pattern_table[i, index])

        code_pattern = np.ravel(code_list)
        return code_pattern

    def coding(self, X, low=0, high=1):
        """ コーディングとは実数をパターンコードに変換する操作です.

        入力値の値域を引数low,highによって設定することができます.

        入力値がlowよりも小さい場合,入力値がlowとした場合のコードパターンを出力します.

        入力値がhighよりも大きい場合,入力値がhighとした場合のコードパターンを出力します.

        Parameters
        ----------
        X : float or ndarray, shape = (input_dim,) or (sample_num,input_dim)
            入力データ
        low : int or float or ndarray, shape =(input_dim,), optional
            入力値の下限
        high : int or float or ndarray, shape =(input_dim,), optional
            入力値の上限

        Returns
        -------
        code_pattern : ndarray, shape =(code_pattern_dim * input_dim,) or (sample_num, input_dim * code_pattern_dim)
            コードパターン

            返り値は各入力次元に対応するコードパターンが結合された1次元ndarrayです.

        """

        # 入力データが0次元(スカラー)の場合
        if X.ndim == 0:
            scaled_element = (X - low) / (high - low)
            index = int(np.floor(scaled_element * self.input_division_num))
            # 正規化
            if index < 0:
                index = 0
            if index > self.input_division_num - 1:
                index = self.input_division_num - 1

            return self.code_pattern_table[0, index]

        elif X.ndim == 1:
            code_pattern = self._real_to_code(X, low, high)
            return code_pattern

        # 入力データが2次元の場合
        elif X.ndim == 2:
            pattern_list = []

            for x in X:
                code_pattern = self._real_to_code(x, low, high)
                pattern_list.append(code_pattern)

            return np.array(pattern_list)
        else:
            raise ValueError('input data dimensions must be 1d or 2d')


class SelectiveDesensitization(PatternCoding):
    """ SelectiveDesensitizationクラスは実数と選択的不感化されたコードパターンの対応関係の管理を行うクラスです.

    選択的不感化とは二つのパターンコード :math:`\mathbf{p}_{i}` , :math:`\mathbf{p}_{j}` を以下の式に従って変換を行う操作です.

    .. math::

        \mathbf{p}_{i,j} = \\frac{\mathbf{p}_{i} + 1}{2} \mathbf{p}_{j}

        \mathbf{p}_{j,i} = \\frac{\mathbf{p}_{j} + 1}{2} \mathbf{p}_{i}

    入力次元がN次元の場合, :math:`\\frac{N(N-1)}{2}` 個の選択的不感化されたコードパターンが生成されます.


    Parameters
    ----------
    code_pattern_dim : int
        コードパターンベクトルの次元数 n
    input_division_num : int
        実数の分割数 q
    reversal_num : int
        反転数 r
    input_dim : int
        入力データの次元数 N
    """

    def __init__(self, binary_vector_dim, division_num, reversal_num, input_dim):
        super().__init__(binary_vector_dim, division_num, reversal_num, input_dim)

    @staticmethod
    def _pattern_to_sd_pattern(code_pattern1, code_pattern2):
        """ pattern2を用いてpattern1の不感化を行う

        Parameters
        ----------
        pattern1 : ndarray, shape = (code_pattern_dim,)
            subject pattern
        pattern2 : ndarray, shape =  (code_pattern_dim,)
            context pattern

        Returns
        -------
        sp_pattern : ndarray, shape = (code_pattern_dim,)
            pattern1 selective desensitizated by pattern2

        """
        sd_code_pattern = (1 + code_pattern1) * code_pattern2 / 2.0
        return sd_code_pattern

    def _real_to_sd_code(self, x, low, high):
        """ 実数xをコードパターンに変換する

        Parameters
        ----------
        x : ndarray, shape = (input_dim,)
        入力値
        low : int or float or ndarray, shape =(input_dim,), optional
        入力値の下限
        high : int or float or ndarray, shape =(input_dim,), optional
        入力値の上限

        Returns
        -------
        sd_code_pattern : ndarray, shape = (input_dim * code_pattern_dim,)
        """

        if isinstance(low, float) or isinstance(low, int):
            low_array = np.ones(self.input_dim) * low
        else:
            low_array = low

        if isinstance(high, float) or isinstance(high, int):
            high_array = np.ones(self.input_dim) * high
        else:
            high_array = high

        code_list = []
        for i, element in enumerate(x):
            scaled_element = (element - low_array[i]) / (high_array[i] - low_array[i])

            index = int(np.floor(scaled_element * self.input_division_num))

            # 正規化
            if index < 0:
                index = 0
            if index > self.input_division_num - 1:
                index = self.input_division_num - 1

            code_list.append(self.code_pattern_table[i, index])

        code_pattern = np.ravel(code_list)
        return code_pattern

    def coding(self, X, low=0, high=1):
        """ 選択的不感化されたコードパターンを出力する

        Parameters
        ----------
        X : ndarray, shape = (input_dim,) or (sample_num,input_dim)
            入力データ
        low : int or float or ndarray, shape =(input_dim,), optional
            入力値の下限
        high : int or float or ndarray, shape =(input_dim,), optional
            入力値の上限

        Returns
        -------
        sd_code_pattern : ndarray, shape = (n_features * (n_features - 1) / 2 * binary_vector_dim,)
        """

        # 入力データが1次元の場合
        if X.ndim == 1:
            code_pattern = self._real_to_code(X, low, high)
            return code_pattern

        # 入力データが2次元の場合
        elif X.ndim == 2:
            pattern_list = []

            for x in X:
                code_pattern = self._real_to_code(x, low, high)
                pattern_list.append(code_pattern)
            return np.array(pattern_list)
        else:
            raise ValueError('input data dimensions must be 1d or 2d')

    def num_to_sd_pattern(X):
        """
        選択的不感化したパターンを返す

        Parameters
        ----------
        X : array, shape = (n_features,)

        Returns
        -------
        sd_code_pattern : ndarray, shape = (n_features * (n_features - 1) / 2 * binary_vector_dim,)
        """
        sd_pattern_list = []
        for x in X:
            pattern_list = []
            for i, element in enumerate(x):
                index = int(np.floor(element * self.division_num))
                pattern_list.append(self.code_pattern_table[i][index])
            sd_pattern = []
            for i, pattern1 in enumerate(pattern_list):
                for j, pattern2 in enumerate(pattern_list):
                    if i == j:
                        continue
                    else:
                        sd_pattern.append(self.pattern_to_sd_pattern(pattern1, pattern2))

            sd_pattern_list.append(np.ravel(sd_pattern))
        return np.array(sd_pattern_list)
