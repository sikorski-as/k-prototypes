import numpy as np
from itertools import zip_longest

np.seterr(all='raise')


class Model:
    def __init__(self, k: int, gamma: float):
        """
        Class for K-prototypes clustering.

        :param k: number of clusters
        :param gamma: scaling parameter of nominal data distance
        """
        self._k: int = k
        self._gamma: float = gamma
        self._n_numerical: int = None
        self._n_nominal: int = None
        self._centers: tuple = None

    def _init_centers(self, numerical: np.ndarray = None, nominal: np.ndarray = None):
        n_rows = numerical.shape[0]

        _, n_numerical, _ = self._number_of_features(numerical, nominal)

        numerical_max = np.max(numerical, axis=0)
        numerical_min = np.min(numerical, axis=0)
        numerical_centers = np.random.uniform(low=0.0, high=1.0, size=(self._k, n_numerical))
        numerical_centers = numerical_centers * (numerical_max - numerical_min) + numerical_min

        nominal_centers = None if nominal is None else nominal[np.random.randint(n_rows, size=self._k)]

        return numerical_centers, nominal_centers

    def _is_valid(self, numerical: np.ndarray, nominal: np.ndarray, raise_exception: bool = False) -> bool:
        n_features, n_numerical, n_nominal = self._number_of_features(numerical, nominal)

        # both arrays should have the same number of rows
        n_rows_valid = True if (self._n_numerical is None or self._n_nominal is None) \
            else n_numerical.shape[0] == n_nominal.shape[1]

        # if model is already initialized,
        # number of model's attributes of both types should be the same as in the given data
        n_numerical_valid = True if self._n_numerical is None else n_numerical == self._n_numerical
        n_nominal_valid = True if self._n_nominal is None else n_nominal == self._n_nominal

        # there should be at least one feature available for clustering
        at_least_one_attribute = n_features > 0

        valid = all([n_rows_valid, n_numerical_valid, n_nominal_valid, at_least_one_attribute])
        if not valid and raise_exception:
            raise ValueError('Provided data is not valid')
        return valid

    def fit(self, numerical: np.ndarray = None, nominal: np.ndarray = None, iterations: int = None):
        """
        Finds K clusters for given numerical and nominal data.

        :param numerical: columns with numerical attributes
        :param nominal: columns with nominal attributes
        :param iterations: number of iterations, None for no limit.
        """
        self._is_valid(numerical, nominal, raise_exception=True)
        if self._centers is None:
            self._centers = self._init_centers(numerical, nominal)

        assignments = None
        i = 0
        iterations = -1 if iterations is None or iterations < 1 else iterations
        while i < iterations:
            i += 1
            numerical_centers, nominal_centers = self._centers
            numerical_distances = np.array([np.linalg.norm(numerical - c, axis=1) for c in numerical_centers])
            nominal_distances = 0 if nominal_centers is None \
                else np.array([np.count_nonzero(nominal != c, axis=1) for c in nominal_centers])
            distances = numerical_distances + self._gamma * nominal_distances
            new_assignments = np.argmin(distances, axis=0)

            if (assignments == new_assignments).all():
                return (numerical_centers, nominal_centers), assignments
            else:
                assignments = new_assignments
                for c in range(self._k):
                    if numerical_centers is not None:
                        numerical_centers[c] = np.mean(numerical[assignments == c], axis=0)
                    if nominal_centers is not None:
                        # this takes the first mode if there are more than one
                        nominal_centers = np.apply_along_axis(
                            lambda x: np.bincount(x).argmax(),
                            axis=0, arr=nominal[assignments == c]
                        )
                    self._centers = numerical_centers, nominal_centers

        return assignments

    def predict(self, numerical: np.ndarray = None, nominal: np.ndarray = None) -> np.ndarray:
        if self._centers is None:
            raise RuntimeError('Model needs to be fitted first')
        else:
            pass

    @property
    def centers_internal(self):
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
        Returns tuple with number of all attributes, number of numerical attributes, number of nominal attributes.
        :return: tuple with numbers of features
        """
        n_numerical = numerical.shape[1] if numerical is not None else 0
        n_nominal = nominal.shape[1] if nominal is not None else 0
        return n_numerical + n_nominal, n_numerical, n_nominal

    @classmethod
    def load_from_file(cls, filepath):
        pass
