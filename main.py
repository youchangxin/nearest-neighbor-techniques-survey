import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
from algorithm.KNN import KNearestNeighbors
from algorithm.wKNN import Weighted_KNN
from algorithm.CNN import Condensed_NN
from algorithm.KD_Tree import KDTree
from algorithm.Ball_Tree import BallTree


def benchmark(dataset, K=5):
    x_train, y_train, x_test, y_test = utils.load_data(dataset)
    result = pd.DataFrame()

    models = [KNearestNeighbors, Weighted_KNN, Condensed_NN, KDTree, BallTree]
    for nearest_neighbor in models:

        model = nearest_neighbor(x_train, y_train, k_neighbors=K)
        acc, time, algorithm = model.score(x_test, y_test)

        new = pd.DataFrame({'Accuracy': acc, 'Time': time}, index=[algorithm])
        result = result.append(new)

    result.to_csv("output/" + dataset + ".csv")


if __name__ == '__main__':
    dataset = ["IJCNN1", "SHUTTLE", "MNIST"]
    for data in dataset:
        benchmark(dataset=data, K=5)


