import numpy as np
import utilities
import matplotlib.pyplot as plt

from multi_layer_perceptron import MultiLayerPerceptronRegression
from sdnn import SDNN
from parallel_perceptron import ParallelPerceptron


def test_model(clf, n_samples=2000):
    test_function = utilities.nonaka
    # test_function = utilities.linear
    # test_function = utilities.cylinder

    train_X = np.random.uniform(0, 1, size=[n_samples, 2])
    train_y = test_function(train_X)
    N = 100
    xx, yy = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    test_X = np.c_[np.ravel(xx), np.ravel(yy)]
    test_y = test_function(test_X)

    clf.fit(train_X, train_y)

    print(clf.score(test_X, test_y))
    prediction_y = clf.predict(test_X)
    plt.pcolor(xx, yy, prediction_y.reshape([N, N]), vmin=-0.2, vmax=1.2)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # clf = MultiLayerPerceptronRegression()
    # clf = SDNN()
    clf = ParallelPerceptron()
    test_model(clf)
