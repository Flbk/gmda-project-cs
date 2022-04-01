import numpy as np
from sklearn.neighbors import NearestNeighbors


def kmeans_pp_init(X, n_clusters, p=2, seed=None):

    rng = np.random.default_rng(seed=seed)
    n, p = X.shape
    indices = np.arange(n)

    centers_array = np.zeros((n_clusters, p))
    weights = np.ones(n) / n

    for i in range(n_clusters):
        center_ind = rng.choice(indices, p=weights)
        center = X[center_ind]
        centers_array[i] = center

        neighs = NearestNeighbors(n_neighbors=1, p=p)
        neighs.fit(centers_array[: i + 1])
        samples_distance, _ = neighs.kneighbors(X=X)
        weights = samples_distance / samples_distance.sum()
        weights = weights.reshape(-1)

    return centers_array


def random_init(X, n_clusters, p=2, seed=0):

    rng = np.random.default_rng(seed=seed)
    n, p = X.shape
    indices = np.arange(n)

    random_ind = rng.choice(indices, n_clusters, replace=False)
    return X[random_ind]

    return centers_array
