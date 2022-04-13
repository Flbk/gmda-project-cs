from typing import Iterable, Optional

import numpy as np
from joblib import Parallel, delayed

from dataset import gaussian_blobs
from kmeans import (KMeans, distance_to_centers, kmeans_pp_init, phi,
                    random_init)


def kmeans_and_phi(
    samples: np.ndarray,
    init: str = "kmeanspp",
    seed: Optional[int] = None,
    normalize: bool = True,
    n_clusters: int = 5,
):
    """Run KMeans algorithm and compute the value of the potential $\phi$.

    Args:
        samples (np.ndarray): The dataset to cluster.
        init (str, optional): Which initialization to use. See KMeans documentation for details. Defaults to "kmeanspp".
        seed (Optional[int], optional): Random seed to use. None means no control (and no reproducibility). Defaults to None.
        normalize (bool, optional): Whether to compute $\phi$ with a sum or mean. Defaults to True.
        n_clusters (int, optional): The number of expected clusters. Defaults to 5.

    Returns:
        float: The value of the $\phi$ on the ran clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, seed=seed, init=init)
    kmeans.fit(samples)
    centers = kmeans.centers
    return phi(samples=samples, centers=centers, normalize=normalize)


def get_n_outliers(
    means: np.ndarray,
    list_covariance_matrices: Iterable[np.ndarray],
    centers: np.ndarray,
    threshold: float = 4.0,
):
    """Compute the number of outliers among a input list of centers, with respect to a gaussian mixture distribution.
    Basically this function looks for the closest gaussian (in term of Mahalanobis distance), and if this cloest gaussian is too far, the point is labled as outlier.

    Args:
        means (np.ndarray, (N, 2)): The means of the gaussians.
        list_covariance_matrices (Iterable[np.ndarray]): List of the covariance matrices of the gaussian.
        centers (np.ndarray, (M, 2)): The centers among which we want to look for outliers.
        threshold (float, optional): Threshold distance to define that a point is an outlier. Defaults to 4.0.

    Returns:
        int: The number of outliers.
    """
    outliers_dist = distance_to_centers(means, list_covariance_matrices, centers)
    n_outliers = (outliers_dist > threshold).sum()
    return n_outliers


def exponent_experiment(
    list_p: Iterable[int],
    n_experiments: int,
    metric_func: str,
    d: float = 1.5,
    ratio_uniform: float = 0.1,
    n_samples: int = 500,
    rotation: str = "random",
    n_clusters: int = 5,
):
    """Run different experiments by varying only the exponent on KMeans++ initialization.
    It can monitor the effect on the number of outliers and on the final value of $\phi$.

    Args:
        list_p (Iterable[int]): The exponents on which to carry the experiment.
        n_experiments (int): The number of time to launch the experiment with a fixed value of the exponent.
        metric_func (str): The name of experiment to make. Can be one of "outliers", "phi", "kmeans+phi".
        d (float, optional): _description_. Defaults to 1.5.
        ratio_uniform (float, optional): _description_. Defaults to 0.1.
        n_samples (int, optional): _description_. Defaults to 500.
        rotation (str, optional): _description_. Defaults to "random".
        n_clusters (int, optional): _description_. Defaults to 5.
    """

    def make_experiments(p: int, n_experiments: int, metric_func: str):
        """A wrapper to perform parallelism.

        Args:
            p (int): The exponent on which to carry the experiment.
            n_experiments (int): The number of time to launch the experiment with a fixed value of the exponent.
            metric_func (str): The name of experiment to make. Can be one of "outliers", "phi", "kmeans+phi".

        Returns:
            _type_: _description_
        """
        uniform_noise = int(ratio_uniform * n_samples)
        list_metric = []
        for n in range(n_experiments):
            # Draw a dataset for each experiment.
            samples, list_covariance_matrices, means = gaussian_blobs(
                d=d,
                uniform_noise=uniform_noise,
                seed=n,
                n_samples=n_samples,
                return_parameter=True,
                list_angles=rotation,
                n_blobs=n_clusters,
            )

            # Need to handle the case p=0 because Scipy built-in Minkowsky distance impose p>=1.
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

            # Initial value of phi.
            if metric_func == "phi":
                metric = phi(centers=centers, samples=samples)

            # Value of phi on the final clustering.
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
