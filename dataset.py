from itertools import cycle
from typing import Iterable, Optional, Tuple, Union

import numpy as np


def gaussian_blobs(
    d: float = 1.0,
    list_concentrations: Iterable[Iterable[float]] = [[1.0, 0.25]],
    list_angles: Iterable[float] = [0, np.pi / 4, -np.pi / 4],
    n_blobs: int = 5,
    seed: int = 0,
    n_samples: int = 3000,
    max_d: int = 3,
    uniform_noise: Optional[int] = None,
    non_negative: int = True,
    return_parameter=False,
):
    """Makes gaussian blobs parametrized by the angles of rotation, their relative distance and dispersion.

    Args:
        d (float, optional): Relative distance between clusters. Defaults to 1.0.
        list_concentrations (Iterable[Iterable[float]], optional): The 2 eigenvalues of each gaussian covariance matrix. Defaults to [[1.0, 0.25]].
        list_angles (Iterable[float], optional): The rotation of each gaussian. Defaults to [0, np.pi / 4, -np.pi / 4].
        n_blobs (int, optional): The number of gaussians. Defaults to 5.
        seed (int, optional): Seed for pseudo-random generation. Defaults to 0.
        n_samples (int, optional): The number of samples to draw. Defaults to 3000.
        max_d (int, optional): Maximum size of the grid on which to sample the gaussians' centers. Expressed in the relative distance unit. Defaults to 3.
        uniform_noise (Optional[int], optional): Ratio of uniform sample to add on top of the gaussian samples. Defaults to None.
        non_negative (int, optional): Whether to shift the points to make sure they have all positive coordinates. Defaults to True.
        return_parameter (bool, optional): Whether to return the covariance matrices and mean of each gaussian. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[Iterable[np.ndarray], ...]]: The drawn dataset plus the mean and covariance matrices list of the gaussians if return_paramter is True.
    """
    rng = np.random.default_rng(seed=seed)
    pos_1d = np.arange(-max_d, max_d)
    X, Y = np.meshgrid(pos_1d, pos_1d)

    # choose the first coordinates of the centers.
    x_centers = rng.choice(pos_1d, size=n_blobs, replace=False).reshape(-1, 1)
    y_centers = rng.choice(pos_1d, size=n_blobs, replace=False).reshape(-1, 1)

    means = d * np.hstack([x_centers, y_centers])
    if list_angles == "random":
        list_angles = rng.integers(0, 359, (n_blobs,))

    list_covariance_matrices = []
    for vp, angle, i in zip(
        cycle(list_concentrations), cycle(list_angles), range(n_blobs)
    ):
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        # rotate the covariance matrix
        cov_matrix = rotation_matrix @ np.diag(vp) @ rotation_matrix.T

        list_covariance_matrices.append(cov_matrix)

    # Define the number of sample per gaussian.
    n_per_blobs = rng.multinomial(n_samples, np.ones(n_blobs) / n_blobs)

    list_gmm_samples = []

    # Sample the points belonging to each gaussian.
    for i in range(n_blobs):
        n_current_gaussian = n_per_blobs[i]
        gaussian_samples = rng.multivariate_normal(
            means[i], list_covariance_matrices[i], n_current_gaussian
        )
        list_gmm_samples.append(gaussian_samples)

    samples = np.vstack(list_gmm_samples)

    # Add uniform noise.
    if uniform_noise is not None:
        outliers = rng.uniform(
            low=-d * (max_d + 2), high=d * (max_d + 2), size=(uniform_noise, 2)
        )
        samples = np.vstack((samples, outliers))

    # Shift the coordinates to make sure they are positive.
    if non_negative:
        min_samples = samples.min(axis=0)
        samples = samples + 2 * np.abs(min_samples)
        means = means + 2 * np.abs(min_samples)
        assert (samples > 0).all()

    if return_parameter:
        return samples, means, list_covariance_matrices

    return samples
