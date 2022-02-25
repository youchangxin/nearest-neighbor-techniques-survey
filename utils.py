from sklearn.datasets import load_svmlight_file
import time
import pandas as pd
import numpy as np

DATASET = {
    "MNIST": "dataset/mnist/mnist.scale",
    "IJCNN1": "dataset/ijcnn1/ijcnn1",
    "SHUTTLE": "dataset/shuttle/shuttle.scale.tr"
}


def load_data(dataset):
    dataset = dataset.upper()
    if dataset not in DATASET:
        raise ValueError("unrecognized dataset: '%s'" % dataset)

    train_dateset = DATASET[dataset]

    # load train dateset
    train_data = load_svmlight_file(DATASET[dataset])
    x_train = train_data[0].toarray()
    y_train = train_data[1]

    # load test dateset
    test_data = load_svmlight_file(DATASET[dataset] + '.t')
    x_test = test_data[0].toarray()
    y_test = test_data[1]
    print('=' * 6 + "Date loaded" + '=' * 6)
    return x_train, y_train, x_test, y_test


def timit(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        func(*args, **kw)
        end_time = time.time()
        print(' [%s] Function run time is %.2f' % (func.__class__, end_time - start_time))
    return wrapper
