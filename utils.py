import time
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from functools import wraps


DATASET = {
    "MNIST": {
        "TRAIN": "dataset/mnist/mnist.scale",
        "TEST": "dataset/mnist/mnist.scale.t"
    },
    "IJCNN1": {
        "TRAIN": "dataset/ijcnn1/ijcnn1",
        "TEST": "dataset/ijcnn1/ijcnn1.t"
    },
    "SHUTTLE": {
        "TRAIN": "dataset/shuttle/shuttle.scale.tr",
        "TEST": "dataset/shuttle/shuttle.scale.t"
    }
}


def load_data(dataset):
    dataset = dataset.upper()
    if dataset not in DATASET:
        raise ValueError("unrecognized dataset: '%s'" % dataset)

    # load train dateset
    train_data = load_svmlight_file(DATASET[dataset]["TRAIN"])
    x_train = train_data[0].toarray()
    y_train = train_data[1]

    # load test dateset
    test_data = load_svmlight_file(DATASET[dataset]["TEST"])
    x_test = test_data[0].toarray()
    y_test = test_data[1]
    print('=' * 8 + "Date loaded" + '=' * 8)
    return x_train, y_train, x_test, y_test


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        class_name = func.__qualname__.split(".")[0]
        print('%s Accuracy : %.5f %%' % (class_name, res))
        print('Search time : %.2f S' % run_time)
        print("*" * 40 + "\n")
        return res, run_time, class_name
    return wrapper




