import numpy as np
from sklearn.neighbors import NearestNeighbors


def kmeans_pp_init(X, n_clusters, p=2, seed=0):

    rng = np.random.default_rng(seed=seed)
    n, p = X.shape
    indices = np.arange(n)

    centers_array = np.zeros((n_clusters, p))
    weights = np.ones(n) / n

    for i in range(n_clusters):
        center = np.random.choice(indices, p=weights)
        centers_array[i] = center

        neighs = NearestNeighbors(n_neighbors=1, p=p)
        neighs.fit(centers_array)
        samples_distance, _ = neighs.kneighbors(X=X)
        weights = samples_distance / samples_distance.sum()

    return centers_array
