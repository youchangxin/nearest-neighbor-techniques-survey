import numpy as np
import utils
from algorithm.KNN import KNearestNeighbors

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = utils.load_data("IJCNN1")
    model = KNearestNeighbors(x_train, y_train)
    model.score(x_test, y_test)

