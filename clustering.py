import time

import pandas as pd

import sklearn.preprocessing
import sklearn.metrics
import numpy as np

from kproto import KPrototypesModel
from tools import print_array_on_one_line


class ClusteringError(Exception):
    pass


def cluster(*args, **kwargs):
    try:
        return _cluster(*args, **kwargs)
    except Exception as e:
        raise ClusteringError(str(e))


def _clustering_report(predicted_labels, silhouette_values, adjusted_rand_values, accuracy_values, times):
    convert = lambda val: '-' if (val is None or val == []) else str(val)
    convert_np_array = lambda val: '-' if (val is None or val == []) else str(val.tolist())

    def find_max(vals):
        rets = None
        if vals is None or vals == []:
            rets = None, None, None, None
        else:
            argmax = np.argmax(vals)
            rets = vals[argmax], argmax, predicted_labels[argmax], times[argmax]
        return convert(rets[0]), convert(rets[1]), convert_np_array(rets[2]), convert(rets[3])

    silh_val, silh_id, silh_labels, silh_time = find_max(silhouette_values)
    arand_val, arand_id, arand_labels, arand_time = find_max(adjusted_rand_values)
    acc_val, acc_id, acc_labels, acc_time = find_max(accuracy_values)

    report = "Report\n" \
             "All attempts:\n" \
        f"  Silhouette values: {convert(silhouette_values)}\n" \
        f"  Adjusted rand index values: {convert(adjusted_rand_values)}\n" \
        f"  Accuracy values: {accuracy_values}\n" \
        f"  Time values: {times}\n" \
             "Silhouette value: \n" \
        f"  Best value: {silh_val}\n" \
        f"  Best model: {silh_id}\n" \
        f"  Time taken: {silh_time}\n" \
        f"  Predicted labels: {silh_labels}\n" \
             "Adjusted rand index:\n" \
        f"  Best value: {arand_val}\n" \
        f"  Best model: {arand_id}\n" \
        f"  Time taken: {arand_time}\n" \
        f"  Predicted labels: {arand_labels}\n" \
             "Accuracy: \n" \
        f"  Best value: {acc_val}\n" \
        f"  Best model: {acc_id}\n" \
        f"  Time taken: {acc_time}\n" \
        f"  Predicted labels: {acc_labels}\n"
    return report


def _accuracy_score(contTbl):
    return contTbl.max(axis=0).sum() / contTbl.sum()


def _cluster(dataset_filename, k,
             numerical_cols, nominal_cols, true_labels_col,
             alpha, beta,
             standardize_mean, standardize_std, attempts):
    df = pd.read_csv(dataset_filename)

    numerical = df.iloc[:, numerical_cols].to_numpy(dtype=float) if numerical_cols != [] else None
    nominal = df.iloc[:, nominal_cols].to_numpy(dtype=np.object) if nominal_cols != [] else None
    true_labels = None if true_labels_col is None else df.iloc[:, [true_labels_col]].to_numpy(dtype=np.object).ravel()

    numerical = sklearn.preprocessing.scale(numerical, with_mean=standardize_mean, with_std=standardize_std)
    scores = []
    accuracies = []
    models = []
    silhs = []
    predicted_labels = []
    times = []

    for i in range(attempts):
        start = time.time()
        model = KPrototypesModel(k=k, init_method='random-improved')
        models.append(model)
        predicted_labels_for_current = model.fit(numerical=numerical, nominal=nominal, iterations=500)
        end = time.time()
        times.append(end - start)
        predicted_labels_for_current = predicted_labels_for_current.ravel()

        # silhouette metric
        silh = sklearn.metrics.silhouette_score(model.distances_matrix(numerical, nominal),
                                                predicted_labels_for_current,
                                                metric='precomputed')
        silhs.append(silh)

        if true_labels is not None:
            # adjusted rand index
            score = sklearn.metrics.adjusted_rand_score(true_labels, predicted_labels_for_current)
            scores.append(score)

            # accuracy
            contMatrix = sklearn.metrics.cluster.contingency_matrix(true_labels, predicted_labels_for_current)
            accuracy = _accuracy_score(contMatrix)
            accuracies.append(accuracy)
        predicted_labels.append(predicted_labels_for_current)

    with print_array_on_one_line():
        return _clustering_report(predicted_labels, silhs, scores, accuracies, times), models
