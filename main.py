import pandas as pd
import numpy as np
from kproto import Model as KPrototypesModel
from pprint import pprint


def numerical_only():
    df = pd.read_csv('data/iris.csv')
    numerical = df.iloc[:, 0:4].to_numpy(dtype=float)
    nominal = None

    model = KPrototypesModel(k=3, gamma=1.0)
    labels = model.fit(numerical=numerical, nominal=nominal, iterations=1)
    pprint(model.centers)
    pprint(labels)


def nominal_only():
    df = pd.read_csv('data/iris.csv')
    print('data\n', df.iloc[:, [4]])

    numerical = None
    nominal = df.iloc[:, [4]].to_numpy(dtype=np.object)

    model = KPrototypesModel(k=3, gamma=1.0)
    labels = model.fit(numerical=numerical, nominal=nominal, iterations=1)

    print('centers\n', model.centers)
    print('labels\n', labels)


def mixed():
    df = pd.read_csv('data/iris.csv')
    print('data\n', df.iloc[:, [4]])

    numerical = df.iloc[:, 0:4].to_numpy(dtype=float)
    nominal = df.iloc[:, [4]].to_numpy(dtype=np.object)

    print('numerical\n', numerical)
    print('nominal\n', nominal)


    model = KPrototypesModel(k=3, gamma=1.0)
    labels = model.fit(numerical=numerical, nominal=nominal, iterations=500)

    print('centers\n', model.centers)
    print('labels\n', labels)


if __name__ == '__main__':
    mixed()
