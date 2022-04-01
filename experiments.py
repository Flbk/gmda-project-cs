import numpy as np
from joblib import Parallel, delayed
from kmeans import phi, kmeans_pp_init, distance_to_centers, KMeans
from dataset import gaussian_blobs


def kmeans_and_phi(samples, init="kmeanspp", seed=None, normalize=True):
    kmeans = KMeans(n_clusters=5, seed=seed, init=init)
    kmeans.fit(samples)
    centers = kmeans.centers
    return phi(samples=samples, centers=centers, normalize=normalize)


def get_n_outliers(means, list_covariance_matrices, centers, threshold=4):
    outliers_dist = distance_to_centers(means, list_covariance_matrices, centers)
    n_outliers = (outliers_dist > threshold).sum()
    return n_outliers


def exponent_experiment(list_p, n_experiments, metric_func):
    def make_experiments(p, n_experiments, metric_func):
        list_metric = []
        for n in range(n_experiments):
            samples, list_covariance_matrices, means = gaussian_blobs(
                d=1.5, uniform_noise=50, seed=n, n_samples=500, return_parameter=True
            )
            centers = kmeans_pp_init(samples, 5, seed=n, p=p)
            if metric_func == "outliers":
                metric = get_n_outliers(
                    means=means,
                    list_covariance_matrices=list_covariance_matrices,
                    centers=centers,
                )
            if metric_func == "phi":
                metric = phi(centers=centers, samples=samples)
            if metric_func == "kmeans+phi":
                metric = kmeans_and_phi(samples=samples, init=centers)
            list_metric.append(metric)
        return np.array(list_metric)

    p_metric = Parallel(n_jobs=-1)(
        delayed(make_experiments)(p, n_experiments, metric_func) for p in list_p
    )
    return np.array(p_metric)
