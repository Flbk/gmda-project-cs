import numpy as np
from joblib import Parallel, delayed

from dataset import gaussian_blobs
from kmeans import KMeans, distance_to_centers, kmeans_pp_init, phi, random_init


def kmeans_and_phi(samples, init="kmeanspp", seed=None, normalize=True, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, seed=seed, init=init)
    kmeans.fit(samples)
    centers = kmeans.centers
    return phi(samples=samples, centers=centers, normalize=normalize)


def get_n_outliers(means, list_covariance_matrices, centers, threshold=4):
    outliers_dist = distance_to_centers(means, list_covariance_matrices, centers)
    n_outliers = (outliers_dist > threshold).sum()
    return n_outliers


def exponent_experiment(
    list_p,
    n_experiments,
    metric_func,
    d=1.5,
    ratio_uniform=0.1,
    n_samples=500,
    rotation="random",
    n_clusters=5,
):
    def make_experiments(p, n_experiments, metric_func):
        uniform_noise = int(ratio_uniform * n_samples)
        list_metric = []
        for n in range(n_experiments):
            samples, list_covariance_matrices, means = gaussian_blobs(
                d=d,
                uniform_noise=uniform_noise,
                seed=n,
                n_samples=n_samples,
                return_parameter=True,
                list_angles=rotation,
                n_blobs=n_clusters,
            )
            if p == 0.0:
                centers = random_init(samples, n_clusters, seed=n)
            else:
                centers = kmeans_pp_init(samples, n_clusters=n_clusters, seed=n, p=p)
            if metric_func == "outliers":
                metric = get_n_outliers(
                    means=means,
                    list_covariance_matrices=list_covariance_matrices,
                    centers=centers,
                )
            if metric_func == "phi":
                metric = phi(centers=centers, samples=samples)
            if metric_func == "kmeans+phi":
                metric = kmeans_and_phi(
                    samples=samples, init=centers, n_clusters=n_clusters
                )
            list_metric.append(metric)
        return np.array(list_metric)

    out = make_experiments(list_p[1], n_experiments, metric_func)
    p_metric = Parallel(n_jobs=-1)(
        delayed(make_experiments)(p, n_experiments, metric_func) for p in list_p
    )
    return np.array(p_metric)
