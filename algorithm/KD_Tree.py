import numpy as np
from collections import Counter
from utils import timeit


class KDNode:
    def __init__(self, value, label, left, right, depth):
        self.value = value
        self.label = label
        self.left = left
        self.right = right
        self.depth = depth
        self.visit = False


class KDTree:
    def __init__(self, values, labels, k_neighbors=5):
        self.values = values
        self.labels = labels
        if len(self.values) == 0:
            raise Exception('Please input not empty data.')
        self.dims_len = self.values.shape[1]
        self.root = self.build_KDTree()
        self.KNN_result = []
        self.K = k_neighbors

    def build_KDTree(self):
        data = np.column_stack((self.values, self.labels))
        return self.build_KDTree_core(data, 0)

    def dist(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def build_KDTree_core(self, data, depth):
        if len(data) == 0:
            return None
        cuttint_dim = depth % self.dims_len

        data = data[data[:, cuttint_dim].argsort()]
        mid_index = len(data) // 2
        node = KDNode(data[mid_index, :-1], data[mid_index, -1], None, None, depth)
        node.left = self.build_KDTree_core(data[0:mid_index], depth + 1)
        node.right = self.build_KDTree_core(data[mid_index + 1:], depth + 1)
        return node

    def _clean_state(self, node):
        if node is None:
            return
        node.visit = False
        self._clean_state(node.left)
        self._clean_state(node.right)

    def predict(self, target):
        if self.root is None:
            raise Exception('KD-Tree is None.')
        if self.K > len(self.values):
            raise ValueError("K in KNN Must Be Greater Than Lenght of data")
        if len(target) != len(self.root.value):
            raise ValueError("x_test Must Has Same Dimension With x_train")
        self.KNN_result = []
        self._query(self.root, target)
        self._clean_state(self.root)
        y_pred = Counter(node[0].label for node in self.KNN_result).most_common(1)[0][0]
        return y_pred

    def _query(self, node, target):
        if node is None or node.visit:
            return
        cur_data = node.value
        cuttint_dim = node.depth % self.dims_len

        if target[cuttint_dim] < cur_data[cuttint_dim]:
            self._query(node.left, target)
        else:
            self._query(node.right, target)

        node.visit = True
        distance = self.dist(cur_data, target)
        if len(self.KNN_result) < self.K:
            self.KNN_result.append((node, distance))

        if distance < self.KNN_result[0][1]:
            self.KNN_result = self.KNN_result[1:] + [(node, distance)]
        self.KNN_result = sorted(self.KNN_result, key=lambda x: -x[1])

        if abs(target[cuttint_dim] - cur_data[cuttint_dim]) < self.KNN_result[0][1] or len(self.KNN_result) < self.K:
            # 在当前切分维度上,以target为中心,最近距离为半径的超体小球如果和该维度上的超平面有交集,那么说明可能还存在更近的数据点
            self._query(node.left, target)
            self._query(node.right, target)

    @timeit
    def score(self, x_test, y_test):
        y_pred = [self.predict(point) for point in x_test]
        return float(sum(y_pred == y_test)) / float(len(y_test))
