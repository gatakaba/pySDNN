# coding:utf-8
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
from base import BaseEstimator
from adaptive_sdnn import AdaptiveSDNN as SDNN


class EnsembleSDNN(BaseEstimator):
    def __init__(self, clf=SDNN, n_estimators=10, sample_rate=0.2, feature_rate=0.3, weight=True, verbose=False):

        self.n_estimators = n_estimators
        self.sample_rate = sample_rate
        self.feature_rate = feature_rate
        self.weight = weight
        self.verbose = verbose

        self.clf_list = []
        self.feature_list = []
        self.score_list = []
        for i in range(self.n_estimators):
            self.clf_list.append(clf())

    def fit(self, X, y):
        self.X_train_, self.y_train_ = np.copy(X), np.copy(y)
        n_samples, n_features = self.X_train_.shape

        for i, clf in enumerate(self.clf_list):
            # feature_index = np.random.permutation(range(n_features))[:int(n_features * self.feature_rate)]
            feature_index = np.random.permutation(range(n_features))[:6]

            indexes = np.random.permutation(range(n_samples))
            indexes_fit = indexes[:int(n_samples * self.sample_rate)]
            indexes_validation = indexes[:int(n_samples * (1 - self.sample_rate))]

            clf.fit(X[indexes_fit, :][:, feature_index], y[indexes_fit])
            self.feature_list.append(np.copy(feature_index))

            if self.weight:
                self.score = clf.score(X[indexes_validation, :][:, feature_index], y[indexes_validation])
                self.score_list.append(self.score)
            if self.verbose:
                if self.weight:
                    print(i, self.score)
                else:
                    print(i, clf.score(X[indexes_validation, :][:, feature_index], y[indexes_validation]))
        return self

    def predict(self, X):
        prediction_list = []
        importance = np.array(self.score_list) ** -1

        normed_importance = importance / np.sum(importance)
        for i in range(self.n_estimators):
            feature_index = self.feature_list[i]
            prediction_list.append(self.clf_list[i].predict(X[:, feature_index]))
        prediction_list = np.array(prediction_list).T
        prediction = np.dot(prediction_list, normed_importance)

        # prediction = np.mean(prediction_list, axis=1)
        return prediction
