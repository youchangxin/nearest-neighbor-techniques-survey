from algorithm.KNN import KNearestNeighbors
from utils import timeit


class Weighted_KNN(KNearestNeighbors):
    def _weight(self, pred_list, dists):
        weights = 1 / (dists + 1e-9)
        y_pred = []
        for idx in range(len(weights)):
            label_set = set(pred_list[idx])
            # using dict to store weight and init the dict
            prob_dic = {label: 0 for label in label_set}
            for label, value in zip(pred_list[idx], weights[idx]):
                prob_dic[label] += value
            # sort the predicted values and return biggest weighted result
            res = max(prob_dic.items(), key=lambda x: x[1])[0]
            y_pred.append(res)
        return y_pred

    @timeit
    def score(self, x_test, y_test):
        pred_list, dists = self.kneighbors(x_test, return_distance=True)
        y_pred = self._weight(pred_list, dists)
        return float(sum(y_pred == y_test)) / float(len(y_test))
