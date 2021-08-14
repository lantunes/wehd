import numpy as np


def cluster_variance(X_indices, D):
    medoid_idx = _medoid(X_indices, D)
    return np.mean([(D[i][medoid_idx])**2 for i in X_indices])


def _medoid(X_indices, D):
    min_sum = None
    min_index = None
    for i in X_indices:
        tot = 0
        for j in X_indices:
            if i == j:
                continue
            dist = D[i][j]
            tot += dist
        if min_sum is None or tot < min_sum:
            min_sum = tot
            min_index = i
    return min_index
