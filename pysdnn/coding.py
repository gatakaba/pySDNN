# coding:utf-8
import numpy as np


class PatternCoding(object):
    """PatternCodingは実数とバイナリベクトルの対応関係を管理する

    パターンは以下の条件を満たす
    * 重複が無い
    * -1と1の数の数が等しい
    * 入力次元ごとに異なるパターンを持つ
    """

    def __init__(self, binary_vector_dim, division_num, reversal_num=1):
        """
        Parameters
        ----------
        binary_vector_dim : int
            バイナリベクトルの次元数
        division_num : int
            実数の分割数
        reversal_num : int
            反転数
        """
        self.binary_vector_dim = binary_vector_dim
        self.division_num = division_num
        self.reversal_num = reversal_num

        self.feature_dim = None
        self.binary_vector_table = None

    def _make_binary_vector_table_1d(self):
        """バイナリべクトルが格納されているテーブルを作成

        Returns
        -------
        binary_vector_table : ndarray, shape = (division_num,binary_vector_dim)

        """

        binary_vector_table = []
        binary_vector = np.ones(self.binary_vector_dim)
        binary_vector[:int(self.binary_vector_dim / 2)] = -1
        np.random.shuffle(binary_vector)
        binary_vector_table.append(binary_vector)

        while len(binary_vector_table) < self.division_num:
            while True:
                tmp_binary_vector = np.copy(binary_vector_table[-1])
                # select reverse index
                index1 = list(
                    np.random.choice(np.where(tmp_binary_vector == -1)[0], size=self.reversal_num, replace=False))
                index2 = list(
                    np.random.choice(np.where(tmp_binary_vector == 1)[0], size=self.reversal_num, replace=False))
                index = index1 + index2

                # reverse selected index
                tmp_binary_vector[index] *= -1

                # if tmp_binary_vector is included in binary_vector_table, add to binary_vector_table
                if not any((tmp_binary_vector == x).all() for x in binary_vector_table):
                    binary_vector_table.append(tmp_binary_vector)
                    break

        return np.array(binary_vector_table)

    def make_binary_vector_table(self, feature_dim):
        """make binary vector table

        binary_vector_table : ndarray, shape = (feature_dim,division_num ,binary_vector_dim)

        Parameters
        ----------
        feature_dim : int
            input feature dimension

        Returns
        -------
        self : returns an instance of self
        """

        self.feature_dim = feature_dim
        self.binary_vector_table = []
        for i in range(self.feature_dim):
            self.binary_vector_table.append(self._make_binary_vector_table_1d())
        self.binary_vector_table = np.stack(self.binary_vector_table)

        return self

    def num_to_pattern(self, X):
        """
        convert real number to binary vector

        Parameters
        ----------
        X : ndarray

        Returns
        -------
        out : ndarray, shape =(binary_vector_dim * feature_dim),(binary_vector_dim * feature_dim,input_data_num)

        """

        if X.ndim == 1:
            pattern_list = []
            for feature_index, element in enumerate(X):
                index = int(np.floor(element * self.division_num))

                pattern_list.append(self.binary_vector_table[feature_index, index])

            return np.ravel(pattern_list)

        elif X.ndim == 2:
            matrix_list = []
            for x in X:
                pattern_list = []
                for i, element in enumerate(x):
                    index = int(np.floor(element * self.division_num))

                    pattern_list.append(self.binary_vector_tables[i][index])
                matrix_list.append(np.ravel(pattern_list))
            return np.array(matrix_list)
        else:
            raise ValueError('input data dimensions must be 1d or 2d')


class SelectiveDesensitization(PatternCoding):
    """
    パターンを選択的不感化する
    """

    def __init__(self, binary_vector_dim, division_num, reversal_num=1):
        super().__init__(binary_vector_dim, division_num, reversal_num)

    def pattern_to_sd_pattern(self, pattern1, pattern2):
        sd_pattern = (1 + pattern1) * pattern2 / 2.0
        return sd_pattern

    def num_to_sd_pattern(self, X):
        """
        選択的不感化したパターンを返す
        Parameters
        ----------
        X : array, shape = (n_features,)

        Returns
        -------
        sd_pattern : ndarray, shape = (n_features * (n_features - 1) / 2 * binary_vector_dim,)
        """
        sd_pattern_list = []
        for x in X:
            pattern_list = []
            for i, element in enumerate(x):
                index = int(np.floor(element * self.division_num))
                pattern_list.append(self.binary_vector_tables[i][index])
            sd_pattern = []
            for i, pattern1 in enumerate(pattern_list):
                for j, pattern2 in enumerate(pattern_list):
                    if i == j:
                        continue
                    else:
                        sd_pattern.append(self.pattern_to_sd_pattern(pattern1, pattern2))

            sd_pattern_list.append(np.ravel(sd_pattern))
        return np.array(sd_pattern_list)
