import numpy as np
from collections import Counter
from utils import timeit


class KNearestNeighbors:
    def __init__(self, x_train, y_train, k_neighbors=5):
        self.x_train = x_train
        self.y_train = y_train
        self.k_neighbors = k_neighbors

    def _distance_metric(self, a, b):
        return np.sqrt(np.sum(np.power((a - b), 2), axis=1))

    def kneighbors(self, x_test, return_distance=False):
        dist = []
        perd = []

        # computing the distance between query and training dataset
        point_dist = [self._distance_metric(x_test, self.x_train) for x_test in x_test]
        for row in point_dist:
            enum_neigh = enumerate(row)  # obtain the index
            # sort by the distance and select top k result
            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:self.k_neighbors]
            pred_list = [self.y_train[pair[0]] for pair in sorted_neigh]  # according to the index get the label
            dist_list = [pair[1] for pair in sorted_neigh]
            dist.append(dist_list)
            perd.append(pred_list)
        if return_distance:
            return perd, np.array(dist)
        return perd

    def predict(self, perd_list):
        y_pred = np.array([Counter(perd).most_common(1)[0][0] for perd in perd_list])
        return y_pred

    @timeit
    def score(self, x_test, y_test):
        perd_list = self.kneighbors(x_test)
        y_pred = self.predict(perd_list)
        return float(sum(y_pred == y_test)) / float(len(y_test))
