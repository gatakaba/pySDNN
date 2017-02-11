# coding:utf-8
import numpy as np


class PatternCoding(object):
    """PatternCodingは実数とバイナリベクトルの対応関係を管理する

    パターンは以下の条件を満たす
    * 重複が無い
    * -1と1が均等になる
    * 次元ごとに異なるパターンを持つ
    """

    def __init__(self, dim_binary_vector, n_division, n_reversal=1):
        """
        Parameters
        ----------
        binary_vector_dim :int
            バイナリベクトルの次元数
        division_num
            実数の分割数
        reversal_num
            反転数
        """
        self.binary_vector_dim = dim_binary_vector
        self.division_num = n_division
        self.reversal_num = n_reversal

    def _make_binary_vector_table(self):
        """make binary vector table

        Returns
        -------
        binary_vector_table : list
           listの中身は-1,1の1d array
        """
        # テーブルの初期化
        binary_vector_table = []
        binary_vector = np.ones(self.binary_vector_dim)
        binary_vector[:int(self.binary_vector_dim / 2)] = -1
        np.random.shuffle(binary_vector)
        binary_vector_table.append(binary_vector)

        # テーブルの作成
        for i in range(self.division_num):
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
                if not any((tmp_binary_vector == x).all() for x in binary_vector_table):
                    # if tmp_binary_vector is included in binary_vector_table, add to binary_vector_table
                    binary_vector_table.append(tmp_binary_vector)
                    break
        return binary_vector_table

    def make_binary_vector_tables(self, train_X):
        self.n_features = train_X.shape[1]
        self.binary_vector_tables = []
        self.cdf_list = []
        self.scale_list = []

        # 実数とインデックスの対応関係を算出
        for i in range(self.n_features):
            hist, bin_edges = np.histogram(train_X[:, i], bins=self.division_num)
            cdf = np.cumsum(hist) / np.max(np.cumsum(hist))
            scale = (bin_edges[1:] + bin_edges[:-1]) / 2.0
            self.cdf_list.append(cdf)
            self.scale_list.append(scale)

        # 各特徴量に対し、パターンベクトルを作成
        for i in range(self.n_features):
            self.binary_vector_tables.append(self._make_binary_vector_table())
        return self

    def _vec_to_pattern(self, x):
        def find_nearest_index(x, x0):
            "Element in nd array `x` closest to the scalar value `x0`"
            idx = np.abs(x - x0).argmin(0)
            return idx

        pattern_list = []
        for i in range(self.n_features):
            cdf_index = find_nearest_index(self.scale_list[i], x[i])
            index = int(round(self.cdf_list[i][cdf_index] * self.division_num))
            pattern_list.append(self.binary_vector_tables[i][index])
        return pattern_list

    def convert_to_pattern(self, X):
        """
        convert real number to binary vector

        Parameters
        ----------
        X : ndarray

        Returns
        -------
        out : ndarray

        TODO : サイズのチェック 例外発生
        """

        if X.ndim == 1:
            return np.ravel(self._vec_to_pattern(X))
        elif X.ndim == 2:
            pattern_list = []
            for x in X:
                pattern_list.append(np.ravel(self._vec_to_pattern(x)))
            return np.array(pattern_list)
        else:
            return None


class SelectiveDesensitization(PatternCoding):
    """
    パターンを選択的不感化する
    """

    def __init__(self, binary_vector_dim, division_num, reversal_num=1):
        super().__init__(binary_vector_dim, division_num, reversal_num)

    def pattern_to_sd_pattern(self, pattern1, pattern2):
        sd_pattern = (1 + pattern1) * pattern2 / 2.0
        return sd_pattern

    def convert_to_sd_pattern(self, X):
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
            pattern_list = self._vec_to_pattern(x)
            sd_pattern = []
            for i, pattern1 in enumerate(pattern_list):
                for j, pattern2 in enumerate(pattern_list):
                    if i == j:
                        continue
                    else:
                        sd_pattern.append(self.pattern_to_sd_pattern(pattern1, pattern2))

            sd_pattern_list.append(np.ravel(sd_pattern))
        return np.array(sd_pattern_list)


if __name__ == "__main__":
    X = np.random.uniform(size=[1000, 3])

    pattern_manager = SelectiveDesensitization(binary_vector_dim=100, division_num=100, reversal_num=1)
    pattern_manager.make_binary_vector_tables(X)
    patterned_X = pattern_manager.convert_to_sd_pattern(X)
    print(patterned_X.shape)
