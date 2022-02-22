import numpy as np
import time
import utils
from algorithm.KNN import KNearestNeighbors
from algorithm.Ball_Tree import BallTree
from algorithm.KD_Tree import KDTree

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = utils.load_data("IJCNN1")
    # y_test = np.expand_dims(y_test, axis=1)
    # test_data = np.concatenate([x_test, y_test], axis=1)
    # np.random.shuffle(test_data)
    # x_test = test_data[:1000, :-1]
    # y_test = test_data[:1000, -1:]
    # y_test = np.squeeze(y_test)

    model = KDTree(x_train, y_train)

    start1 = time.time()
    model.score(x_test, y_test)
    end1 = time.time()
    print("Search time: ", end1-start1)

    model = KNearestNeighbors(x_train, y_train)
    start1 = time.time()
    model.score(x_test, y_test)
    end1 = time.time()
    print("Search time: ", end1 - start1)

    # from sklearn.neighbors import KNeighborsClassifier
    # model = KNeighborsClassifier(algorithm="brute", n_jobs=1)
    # model.fit(x_train, y_train)
    # start1 = time.time()
    # res = model.score(x_test, y_test)
    # end1 = time.time()
    # print("SKlearn kd-tree accuracy: ", res)
    # print("Search time: ", end1 - start1)

    model = BallTree(x_train, y_train)

    start1 = time.time()
    model.score(x_test, y_test)
    end1 = time.time()
    print("Search time: ", end1 - start1)





