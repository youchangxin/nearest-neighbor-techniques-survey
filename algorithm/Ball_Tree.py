import numpy as np
from collections import Counter
from utils import timeit


class Ball:
    def __init__(self, center, radius, points, left, right):
        self.center = center  # 使用该点即为球中心,而不去精确地去找最小外包圆的中心
        self.radius = radius
        self.left = left
        self.right = right
        self.points = points


class BallTree:
    def __init__(self, values, labels, k_neighbors=5):
        self.values = values
        self.labels = labels
        self.K = k_neighbors
        if len(self.values) == 0:
            raise Exception('Please input not empty data.')
        self.root = self.build_BallTree()
        self.KNN_max_now_dist = np.inf
        self.KNN_result = [(None, self.KNN_max_now_dist)]

    def build_BallTree(self):
        data = np.column_stack((self.values, self.labels))
        return self.build_BallTree_core(data)

    def dist(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # data:带标签的数据且已经排好序的
    def build_BallTree_core(self, data):
        if len(data) == 0:
            return None
        if len(data) == 1:
            return Ball(data[0, :-1], 0.001, data, None, None)
        # 当每个数据点完全一样时,全部归为一个球,及时退出递归,不然会导致递归层数太深出现程序崩溃
        data_disloc = np.row_stack((data[1:], data[0]))
        if np.sum(data_disloc - data) == 0:
            return Ball(data[0, :-1], 1e-100, data, None, None)

        cur_center = np.mean(data[:, :-1], axis=0)
        dists_with_center = np.array([self.dist(cur_center, point) for point in data[:, :-1]])
        max_dist_index = np.argmax(dists_with_center)
        max_dist = dists_with_center[max_dist_index]
        root = Ball(cur_center, max_dist, data, None, None)
        point1 = data[max_dist_index]

        dists_with_point1 = np.array([self.dist(point1[:-1], point) for point in data[:, :-1]])
        max_dist_index2 = np.argmax(dists_with_point1)
        point2 = data[max_dist_index2]

        dists_with_point2 = np.array([self.dist(point2[:-1], point) for point in data[:, :-1]])
        assign_point1 = dists_with_point1 < dists_with_point2

        root.left = self.build_BallTree_core(data[assign_point1])
        root.right = self.build_BallTree_core(data[~assign_point1])
        return root  # 是一个Ball

    def search_KNN(self, target):
        if self.root is None:
            raise Exception('KD-Tree Must Be Not empty.')
        if self.K > len(self.values):
            raise ValueError("K in KNN Must Be Greater Than Lenght of data")
        if len(target) != len(self.root.center):
            raise ValueError("Target Must Has Same Dimension With Data")
        self.KNN_result = [(None, self.KNN_max_now_dist)]
        self.nums = 0
        self.search_KNN_core(self.root, target)
        return Counter(node[0][-1] for node in self.KNN_result).most_common(1)[0][0]

    def insert(self, root_ball, target):
        for node in root_ball.points:
            distance = self.dist(target, node[:-1])
            if len(self.KNN_result) < self.K:
                self.KNN_result.append((node, distance))
            elif distance < self.KNN_result[0][1]:
                self.KNN_result = self.KNN_result[1:] + [(node, distance)]
            self.KNN_result = sorted(self.KNN_result, key=lambda x: -x[1])

    # root是一个Ball
    def search_KNN_core(self, root_ball, target):
        if root_ball is None:
            return
        # 在合格的超体空间(必须是最后一层的子空间)内查找更近的数据点
        if root_ball.left is None or root_ball.right is None:
            self.insert(root_ball, target)
        if abs(self.dist(root_ball.center, target)) <= root_ball.radius + self.KNN_result[0][1]:
            self.search_KNN_core(root_ball.left, target)
            self.search_KNN_core(root_ball.right, target)

    @timeit
    def score(self, x_test, y_test):
        y_pred = [self.search_KNN(point) for point in x_test]
        return float(sum(y_pred == y_test)) / float(len(y_test))
