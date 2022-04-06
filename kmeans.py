import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def kmeans_pp_init(X, n_clusters, p=2, seed=None, return_ind=False):

    rng = np.random.default_rng(seed=seed)
    n, m = X.shape
    indices = np.arange(n)
    selected_indices = []
    centers_array = np.zeros((n_clusters, m))
    weights = np.ones(n) / n

    for i in range(n_clusters):
        center_ind = rng.choice(indices, p=weights)
        center = X[center_ind]
        centers_array[i] = center

        neighs = NearestNeighbors(n_neighbors=1, p=p)
        neighs.fit(centers_array[: i + 1])
        samples_distance, ind = neighs.kneighbors(X=X)
        selected_indices.append(ind)
        pow_samples_distance = (samples_distance / samples_distance.max()) ** p
        weights = (pow_samples_distance) / pow_samples_distance.sum()
        weights = weights.reshape(-1)
    if return_ind:
        return centers_array, ind
    return centers_array


def random_init(X, n_clusters, p=2, seed=0):

    rng = np.random.default_rng(seed=seed)
    n, p = X.shape
    indices = np.arange(n)

    random_ind = rng.choice(indices, n_clusters, replace=False)
    return X[random_ind]

    return centers_array


def distance_to_centers(means, list_covariance, centers):
    list_dist = []
    for mean, cov in zip(list_covariance, means):
        dist = cdist(
            centers, mean[None, :], metric="mahalanobis", VI=np.linalg.inv(cov)
        )
        list_dist.append(dist)
    return np.hstack(list_dist).min(axis=1)


def phi(centers, samples, normalize=False):
    neighs = NearestNeighbors(n_neighbors=1, metric="sqeuclidean")
    neighs.fit(centers)
    dist, _ = neighs.kneighbors(samples)
    if normalize:
        return dist.sum() / samples.shape[0]
    return dist.sum()


class KMeans:
    """KMeans clustering."""

    def __init__(self, n_clusters, init="kmeanspp", seed=0, max_iter=100):
        """max_iter (int, optional): maximum number of iterations. Default to 100."""
        self.n_clusters = n_clusters
        self.max_iter = 100
        self.init = init
        self.seed = seed

    def predict_(self, X, centers):
        """Assign each samples of input data to the closest centroid.

        Args:
            X (array (n, p)): input data to assign.
            centers (array (k, p)): the centers. For k in {0, .., K-1} center[k] contains the center of class k.

        Returns:
            array (n,): the labels (the indices of the closest centroid) for each sample.
        """
        neighs = NearestNeighbors(n_neighbors=1)
        neighs.fit(centers)
        _, labels = neighs.kneighbors(X=X)
        return labels.reshape(-1)

    def fit(
        self,
        X,
    ):
        """Fit KMeans to the data.
        The initialisation is done by drawing n_classes random points (without replacement) or with kmeans++.

        Args:
            X (array (n, p)): data to cluster
        """
        if isinstance(self.init, np.ndarray):
            centers = self.init.copy()
        elif self.init == "random":
            # random initialisation of centroids
            centers = random_init(X, self.n_clusters)
        elif self.init == "kmeanspp":
            centers = kmeans_pp_init(X, self.n_clusters, seed=self.seed)

        for _ in range(self.max_iter):
            # Get the closest centroid of each sample.
            labels = self.predict_(X, centers)

            # Update the centroids values for each class.
            for k in range(self.n_clusters):
                class_mean = X[labels == k].mean(axis=0)
                centers[k] = class_mean

        self.centers = centers
        return self

    def predict(self, X):
        """Predict the classes of input data using the fitted centroids.

        Args:
            X (array (n, p)): data.

        Returns:
            array (n,): predicted labels (i.e., indices of the closest centroid).
        """
        return self.predict_(X, self.centers)

    def fit_predict(self, X):
        self.fit(X)
        return self.centers, self.predict(X)
