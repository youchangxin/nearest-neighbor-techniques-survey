import numpy as np
from algorithm.KNN import KNearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from utils import timeit


class Condensed_NN(KNearestNeighbors):

    def __init__(self, x_train, y_train, k_neighbors=5):
        self.x_train = x_train
        self.y_train = y_train
        self.k_neighbors = k_neighbors
        self.x_train, self.y_train = self.get_store()

    def get_store(self):
        num_samples = np.size(self.x_train, 0)
        # Print the initial number of instances in the condensed training set
        print("Before condensing: " + str(num_samples) + " training instances")

        # init the bins STORE and GRABBAG
        x_store = np.array(self.x_train[0, None])
        y_store = np.array(self.y_train[0, None])

        x_grabbage = []
        y_grabbage = []

        # For the second instance to the last instance in the condensed training_set
        for row in range(1, num_samples):
            label = self.y_train[row]

            one_nn = KNeighborsClassifier(n_neighbors=1)
            one_nn.fit(x_store, y_store)
            y_pred = one_nn.predict(self.x_train[row, None])

            # If predict value is not equal to the label
            # Append that instance to the store array
            # Append that instance to the grabbag array
            if y_pred != label:
                x_store = np.concatenate([x_store, self.x_train[row, None]], axis=0)
                y_store = np.concatenate([y_store, self.y_train[row, None]], axis=0)
            else:
                x_grabbage.append(self.x_train[row, None])
                y_grabbage.append(self.y_train[row, None])

        x_grabbage = np.array(x_grabbage)
        y_grabbage = np.array(y_grabbage)

        # Declare the stopping criteria.
        # 1. The GRABBAG is exhausted
        # 2. One complete pass is made through GRABBAG with no transfers to STORE.

        no_more_transfers_to_store = False
        is_garbbag_exhausted = None

        if len(x_grabbage) > 0:
            is_garbbag_exhausted = False
        else:
            is_garbbag_exhausted = True

        while not no_more_transfers_to_store and not is_garbbag_exhausted:
            # Reset the number of transfers_made to 0
            transfers_made = 0

            remove_idx = []
            for idx in range(len(x_grabbage)):
                label = y_grabbage[idx]

                one_nn = KNeighborsClassifier(n_neighbors=1)
                one_nn.fit(x_store, y_store)
                y_pred = one_nn.predict(x_grabbage[idx])
                # If actual class value is not equal to the prediction
                # Append that instance to the store array
                # else that instance to the grabbag array
                if y_pred != label:
                    x_store = np.concatenate([x_store, x_grabbage[idx]], axis=0)
                    y_store = np.concatenate([y_store, y_grabbage[idx]], axis=0)
                    remove_idx.append(idx)
                    transfers_made += 1
            x_grabbage = np.delete(x_grabbage, remove_idx, axis=0)
            y_grabbage = np.delete(y_grabbage, remove_idx, axis=0)

            if len(x_grabbage) > 0:
                is_garbbag_exhausted = False
            else:
                is_garbbag_exhausted = True

            if transfers_made > 0:
                no_more_transfers_to_store = False
            else:
                no_more_transfers_to_store = True

        print("After condensing: " + str(np.size(x_store, 0)) + " training instances")
        return x_store, y_store

    @timeit
    def score(self, x_test, y_test):
        perd_list = self.kneighbors(x_test)
        y_pred = self.predict(perd_list)
        return float(sum(y_pred == y_test)) / float(len(y_test))
