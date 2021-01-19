import sklearn.metrics

import pandas as pd
import numpy as np
from kproto import KPrototypesModel as KPrototypesModel
from pprint import pprint


def numerical_only():
    df = pd.read_csv('data/iris.csv')
    numerical = df.iloc[:, 0:4].to_numpy(dtype=float)
    nominal = None

    model = KPrototypesModel(k=3, beta=1.0)
    labels = model.fit(numerical=numerical, nominal=nominal, iterations=500)
    pprint(model.centers)
    pprint(labels)


def nominal_only():
    df = pd.read_csv('data/iris.csv')
    print('data\n', df.iloc[:, [4]])

    numerical = None
    nominal = df.iloc[:, [4]].to_numpy(dtype=np.object)

    model = KPrototypesModel(k=3, beta=1.0)
    labels = model.fit(numerical=numerical, nominal=nominal, iterations=500)

    print('centers\n', model.centers)
    print('labels\n', labels)


def mixed_flowers():
    df = pd.read_csv('data/iris.csv')
    print('data\n', df.iloc[:, [4]])

    numerical = df.iloc[:, 0:4].to_numpy(dtype=float)
    nominal = df.iloc[:, [4]].to_numpy(dtype=np.object)
    true_labels = df.iloc

    print('numerical\n', numerical)
    print('nominal\n', nominal)

    model = KPrototypesModel(k=3, beta=1.0)
    predicted_labels = model.fit(numerical=numerical, nominal=nominal, iterations=500)
    # print(sklearn.metrics.silhouette_score(dataset, labels))
    print(sklearn.metrics.rand_score(true_labels, predicted_labels))

    print('centers\n', model.centers)
    print('labels\n', predicted_labels)


def test_mixed():
    df = pd.read_csv('data/iris.csv')
    print('data\n', df.iloc[:, [4]])

    numerical = df.iloc[:, 0:4].to_numpy(dtype=float)
    nominal = df.iloc[:, [4]].to_numpy(dtype=np.object)
    true_labels = df.iloc[:, [4]].to_numpy(dtype=np.object)

    # print('numerical\n', numerical)
    # print('nominal\n', nominal)

    model = KPrototypesModel(k=3, beta=1.0)
    predicted_labels = model.fit(numerical=numerical, nominal=None, iterations=500)
    predicted_labels = predicted_labels.ravel()
    # print(sklearn.metrics.silhouette_score(dataset, labels))
    print(sklearn.metrics.rand_score(true_labels.ravel(), predicted_labels))

    print('centers\n', model.centers)
    print('labels\n', predicted_labels)


if __name__ == '__main__':
    test_mixed()
