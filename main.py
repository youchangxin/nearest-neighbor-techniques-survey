import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
from algorithm.KNN import KNearestNeighbors
from algorithm.wKNN import Weighted_KNN
from algorithm.CNN import Condensed_NN
from algorithm.KD_Tree import KDTree
from algorithm.Ball_Tree import BallTree


def benchmark(x_train, y_train, x_test, y_test, dataset, K=5):
    struct_pd = pd.DataFrame()
    unstruct_pd = pd.DataFrame()

    struct_models = [KNearestNeighbors, Weighted_KNN, Condensed_NN]
    for struct_model in struct_models:

        model = struct_model(x_train, y_train, k_neighbors=K)
        acc, time, algorithm = model.score(x_test, y_test)

        new = pd.DataFrame({'Accuracy': acc, 'Time': time}, index=[algorithm])
        struct_pd = struct_pd.append(new)

    unstruct_models = [KDTree, BallTree]
    for unstruct_model in unstruct_models:
        model = unstruct_model(x_train, y_train, k_neighbors=K)
        acc, time, algorithm = model.score(x_test, y_test)

        new = pd.DataFrame({'Accuracy': acc, 'Time': time}, index=[algorithm])
        unstruct_pd = unstruct_pd.append(new)

    struct_pd.to_csv("output/struct_" + dataset + ".csv")
    unstruct_pd.to_csv("output/unstruct_" + dataset + ".csv")


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = utils.load_data("SHUTTLE")
    # y_test = np.expand_dims(y_test, axis=1)
    # test_data = np.concatenate([x_test, y_test], axis=1)
    # np.random.shuffle(test_data)
    # x_test = test_data[:1000, :-1]
    # y_test = test_data[:1000, -1:]
    # y_test = np.squeeze(y_test)

    dataset = ["SHUTTLE", "IJCNN1", "MNIST"]
    for data in dataset:
        benchmark(x_train, y_train, x_test, y_test, dataset=data, K=5)


