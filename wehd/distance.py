from typing import List
from sklearn.metrics import pairwise_distances
import numpy as np


class WEHD:
    """
    A distance for heterogeneous feature vectors, that uses the Euclidean metric for numerical continuous features,
    and the Hamming metric for categorical features.
    """
    def __init__(self, categorical_indices: List[int], weights: List[float]):
        """
        :param categorical_indices: the indices of the feature vector that contain categorical features
        :param weights: the weights for weighted Euclidean and Hamming metrics
        """
        self._cat_ind = categorical_indices
        self._weights = np.array(weights)
        self._n = len(self._weights)
        self._num_ind = [i for i in range(self._n) if i not in categorical_indices]

    def get_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        The general approach is to break up the feature vector matrix into two parts:
          1. the continuous numerical part, that has a Euclidean distance
          2. the categorical part, that has a Hamming distance
        Then, the pairwise distances are computed for each of these separately to create
        the distance matrices D_euclid and D_hamming, where are added together.

        :param X: the list of feature vectors
        :return: the distance matrix
        """
        X_euclid = X[:, self._num_ind]
        X_hamming = X[:, self._cat_ind]

        D_euclid = 0.
        if X_euclid.shape[1] > 0:
            D_euclid = pairwise_distances(X_euclid, metric="minkowski", p=2, w=self._weights[self._num_ind])

        D_hamming = 0.
        if X_hamming.shape[1] > 0:
            D_hamming = pairwise_distances(X_hamming, metric="hamming", w=self._weights[self._cat_ind])

        return D_euclid + D_hamming
