import pandas as pd
from kproto import Model as KPrototypesModel
from pprint import pprint


def main():
    df = pd.read_csv('data/iris.csv')
    numerical = df.iloc[:, 1:4].to_numpy(dtype=float)
    nominal = None

    model = KPrototypesModel(k=3, gamma=1.0)
    labels = model.fit(numerical=numerical, nominal=nominal, iterations=1)
    pprint(model.centers)
    pprint(labels)


if __name__ == '__main__':
    main()
