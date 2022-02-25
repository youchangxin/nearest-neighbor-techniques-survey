import numpy as np
import utils
from algorithm.KNN import KNearestNeighbors
from algorithm.wKNN import Weighted_KNN
from algorithm.CNN import Condensed_NN
from algorithm.KD_Tree import KDTree
from algorithm.Ball_Tree import BallTree


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = utils.load_data("SHUTTLE")
    y_test = np.expand_dims(y_test, axis=1)
    test_data = np.concatenate([x_test, y_test], axis=1)
    np.random.shuffle(test_data)
    x_test = test_data[:10, :-1]
    y_test = test_data[:10, -1:]
    y_test = np.squeeze(y_test)

    model = KNearestNeighbors(x_train, y_train, k_neighbors=1)
    acc, time, _ = model.score(x_test, y_test)

    model = Weighted_KNN(x_train, y_train)
    model.score(x_test, y_test)

    model = Condensed_NN(x_train, y_train)
    model.score(x_test, y_test)

    model = KDTree(x_train, y_train)
    model.score(x_test, y_test)

    model = BallTree(x_train, y_train)
    model.score(x_test, y_test)
