import numpy as np


def euclidian_distance(a, b):
    return np.sqrt(np.sum(np.power((b - a), 2), axis=1))


class KNearestNeighbors:
    def __init__(self, x_train, y_train, k_neighbors=5, n_classes=2):

        self.x_train = x_train
        self.y_train = y_train
        self.k_neighbors = k_neighbors
        self.n_classes = n_classes

    def kneighbors(self, x_test, return_distance=False):
        dist = []
        neigh_idx = []
        point_dist = [euclidian_distance(x_test, self.x_train) for x_test in x_test]
        print("completed distances computation")
        for row in point_dist:
            enum_neigh = enumerate(row)
            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:self.k_neighbors]
            ind_list = [pair[0] for pair in sorted_neigh]
            dist_list = [pair[1] for pair in sorted_neigh]
            dist.append(dist_list)
            neigh_idx.append(ind_list)
        if return_distance:
            return neigh_idx, np.array(dist)
        return neigh_idx

    def predict(self, neigh_ind):
        y_pred = np.array([np.argmax(np.bincount(self.y_train[idx])) for idx in neigh_ind])
        return y_pred

    def score(self, x_test, y_test):
        y_pred = self.kneighbors(x_test)
        res = float(sum(y_pred == y_test)) / float(len(y_test))
        print(res)
        return res
