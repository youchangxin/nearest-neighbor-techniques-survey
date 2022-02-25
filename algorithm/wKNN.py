from algorithm.KNN import KNearestNeighbors
import numpy as np


class WKNN(KNearestNeighbors):
    def _weight(self, pred_list, dists):
        weights = 1 / (dists + 1e-9)
        y_pred = []
        for idx in range(len(weights)):
            label_set = set(pred_list[idx])
            prob_dic = {label: 0 for label in label_set}

            for label, value in zip(pred_list[idx], weights[idx]):
                prob_dic[label] += value
            res = max(prob_dic.items(), key=lambda x: x[1])[0]
            y_pred.append(res)
        return y_pred

    def score(self, x_test, y_test):
        pred_list, dists = self.kneighbors(x_test, return_distance=True)
        y_pred = self._weight(pred_list, dists)
        res = float(sum(y_pred == y_test)) / float(len(y_test))
        print("Weighted KNN accuracy: ", res)
        return res
