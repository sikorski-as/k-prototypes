import random

import numpy as np
from scipy import stats

np.seterr(all='raise')


class KPrototypesModel:
    def __init__(self, k: int, alpha: float = 1.0, beta: float = 1.0, init_method: str = 'random-improved'):
        """
        Class for K-prototypes clustering.

        :param k: number of clusters
        :param alpha: scaling parameter of numerical data distance (default: 1.0)
        :param beta: scaling parameter of nominal data distance (default: 1.0)
        """
        assert init_method in ['random', 'random-improved']
        self._k: int = k
        self._alpha: float = alpha
        self._beta: float = beta
        self._n_numerical: int = None
        self._n_nominal: int = None
        self._centers: tuple = None
        self._init_method = init_method

    def _new_centers(self, numerical: np.ndarray = None, nominal: np.ndarray = None) -> tuple:
        """
        Return newly initialized centers from given data.

        :param numerical: columns with numerical data
        :param nominal: columns with nominal data
        :return: tuple where:
            first element is np.ndarray where rows are numerical attributes of clusters' centers,
            second element is np.ndarray where rows are nominal attributes for clusters' centers
        """
        if self._init_method == 'random':
            return self._new_centers_random(numerical, nominal)
        elif self._init_method == 'random-improved':
            return self._new_centers_random_improved(numerical, nominal)
        else:
            raise RuntimeError('{} init method unknown'.format(self._init_method))

    def _new_centers_random(self, numerical: np.ndarray = None, nominal: np.ndarray = None) -> tuple:
        n_rows = self._number_of_rows(numerical, nominal)
        _, n_numerical, n_nominal = self._number_of_features(numerical, nominal)
        numerical_centers = None
        nominal_centers = None

        if numerical is not None:
            numerical_max = np.max(numerical, axis=0)
            numerical_min = np.min(numerical, axis=0)
            numerical_centers = np.random.uniform(low=0.0, high=1.0, size=(self._k, n_numerical))
            numerical_centers = numerical_centers * (numerical_max - numerical_min) + numerical_min

        if nominal is not None:
            nominal_centers = np.copy(nominal[np.random.randint(n_rows, size=self._k)])

        return numerical_centers, nominal_centers

    def _new_centers_random_improved(self, numerical: np.ndarray = None, nominal: np.ndarray = None) -> tuple:
        n_rows = self._number_of_rows(numerical, nominal)
        n_centers = self._k
        _, n_numerical, n_nominal = self._number_of_features(numerical, nominal)

        numerical_centers = None
        if numerical is not None:
            numerical_max = np.max(numerical, axis=0)
            numerical_min = np.min(numerical, axis=0)
            numerical_centers = np.random.uniform(low=0.0, high=1.0, size=(self._k, n_numerical))
            numerical_centers = numerical_centers * (numerical_max - numerical_min) + numerical_min

        nominal_centers = None
        if nominal is not None:
            nominal_centers = np.ndarray(shape=(n_centers, n_nominal), dtype=np.object)
            for col_id in range(n_nominal):
                pool = np.unique(nominal[:, col_id]).tolist()  # type: list
                vals_for_current_column = random.choices(pool, k=n_centers)
                for center_id in range(n_centers):
                    nominal_centers[center_id, col_id] = vals_for_current_column[center_id]

        return numerical_centers, nominal_centers

    def _is_valid(self, numerical: np.ndarray, nominal: np.ndarray, raise_exception: bool = False) -> bool:
        """
        Checks if given data is valid for K-prototypes model:
        1. Number of rows in numerical and nominal data is the same (if both are given).
        2. If model is already initialized (trained already), new data should have the same number of features.
        3. At least one feature in numerical or nominal data.

        :param numerical:
        :param nominal:
        :param raise_exception: should return exception when data is not valid?
        :return: True if data is valid, False otherwise
        """
        n_features, n_numerical, n_nominal = self._number_of_features(numerical, nominal)

        # 1. both arrays should have the same number of rows
        n_rows_valid = True if (self._n_numerical is None or self._n_nominal is None) \
            else n_numerical.shape[0] == n_nominal.shape[1]

        # 2. if model is already initialized,
        # number of model's attributes of both types should be the same as in the given data
        n_numerical_valid = True if self._n_numerical is None else n_numerical == self._n_numerical
        n_nominal_valid = True if self._n_nominal is None else n_nominal == self._n_nominal

        # 3. there should be at least one feature available for clustering
        at_least_one_attribute = n_features > 0

        valid = all([n_rows_valid, n_numerical_valid, n_nominal_valid, at_least_one_attribute])
        if not valid and raise_exception:
            raise ValueError('Provided data is not valid')
        return valid

    def fit(self, numerical: np.ndarray = None, nominal: np.ndarray = None, iterations: int = 300):
        """
        Finds K clusters for given numerical and nominal data.

        :param numerical: columns with numerical attributes
        :param nominal: columns with nominal attributes
        :param iterations: number of iterations, None for no limit (algorithm stops when there is no change in cluster assignment).
        """
        self._is_valid(numerical, nominal, raise_exception=True)
        if self._centers is None:
            self._centers = self._new_centers(numerical, nominal)

        assignments = None
        i = 0
        iterations = -1 if iterations is None or iterations < 1 else iterations
        while i < iterations:
            i += 1
            numerical_centers, nominal_centers = self._centers

            # numerical distance
            numerical_distances = 0 if numerical_centers is None \
                else np.array([np.linalg.norm(numerical - c, axis=1) for c in numerical_centers])  # L2 norm

            # nominal distance
            nominal_distances = 0 if nominal_centers is None \
                else np.array([np.count_nonzero(nominal != c, axis=1) for c in nominal_centers])

            # composite distance (numerical distance and nominal distance together)
            distances = self._alpha * numerical_distances + self._beta * nominal_distances

            # assign points to new centers
            new_assignments = np.argmin(distances, axis=0)
            self._centers = numerical_centers, nominal_centers

            if (assignments == new_assignments).all():  # nothing changed, algorithm converged
                return assignments
            else:
                assignments = new_assignments
                # adjust cluster centers:
                for c in range(self._k):
                    if numerical_centers is not None:
                        points_assigned_to_current_cluster = numerical[assignments == c, :]
                        cluster_empty = points_assigned_to_current_cluster.shape[0] == 0
                        if not cluster_empty:
                            numerical_centers[c] = np.mean(numerical[assignments == c, :], axis=0)
                        else:  # if cluster is empty, new random center might be chosen, I simply skip it
                            pass
                    if nominal_centers is not None:
                        points_assigned_to_current_cluster = nominal[assignments == c, :]
                        cluster_empty = points_assigned_to_current_cluster.shape[0] == 0
                        if not cluster_empty:
                            # this takes the first mode if there are more than one
                            nominal_centers[c] = stats.mode(points_assigned_to_current_cluster)[0]
                        else:  # if cluster is empty, new random center might be chosen, I simply skip it
                            pass
                self._centers = numerical_centers, nominal_centers

        return assignments

    def predict(self, numerical: np.ndarray = None, nominal: np.ndarray = None) -> np.ndarray:
        if self._centers is None:
            raise RuntimeError('Model needs to be fitted first')
        else:
            pass

    @property
    def centers_internal(self):
        """
        Returns internal representation of clusters' centers
        :return:
        """
        if self._centers is None:
            raise RuntimeError('Model is not fitted')
        else:
            return self._centers

    @property
    def centers(self):
        numerical_parts, nominal_parts = self.centers_internal

        if numerical_parts is None:
            return nominal_parts.tolist()

        if nominal_parts is None:
            return numerical_parts.tolist()

        # assert same number of rows
        assert numerical_parts.shape[0] == nominal_parts.shape[0]
        return np.hstack((numerical_parts, nominal_parts))

    @staticmethod
    def _number_of_features(numerical: np.ndarray, nominal: np.ndarray) -> tuple:
        """
        Returns tuple of size 3 with: number of all attributes, number of numerical attributes, number of nominal attributes.
        :return: tuple with numbers of features
        """
        n_numerical = numerical.shape[1] if numerical is not None else 0
        n_nominal = nominal.shape[1] if nominal is not None else 0
        return n_numerical + n_nominal, n_numerical, n_nominal

    @staticmethod
    def _number_of_rows(numerical: np.ndarray, nominal: np.ndarray):
        num_rows = None if numerical is None else numerical.shape[0]
        nom_rows = None if nominal is None else nominal.shape[0]

        if num_rows is not None and nom_rows is not None:
            assert num_rows == nom_rows
            return num_rows
        elif num_rows is not None:
            return num_rows
        elif nom_rows is not None:
            return nom_rows
        else:
            raise ValueError('Number of rows cannot be inferred when both arrays are None')

    @classmethod
    def load_from_file(cls, filepath):
        raise RuntimeError('Not implemented')

    def save_to_file(self, filepath):
        raise RuntimeError('Not implemented')
