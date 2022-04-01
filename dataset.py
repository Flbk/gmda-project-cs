import numpy as np
from itertools import cycle


def gaussian_blobs(
    d=1.0,
    list_concentrations=[[1.0, 0.25]],
    list_angles=[0, np.pi / 4, -np.pi / 4],
    n_blobs=5,
    seed=0,
    n_samples=3000,
    max_d=3,
    uniform_noise=None,
    return_parameter=False,
):
    """Makes gaussian blobs parametrized by the angles of rotation, their relative distance and dispersion."""
    rng = np.random.default_rng(seed=seed)
    pos_1d = np.arange(-max_d, max_d)
    X, Y = np.meshgrid(pos_1d, pos_1d)

    # choose first coordinates
    x_centers = rng.choice(pos_1d, size=n_blobs, replace=False).reshape(-1, 1)
    y_centers = rng.choice(pos_1d, size=n_blobs, replace=False).reshape(-1, 1)

    means = d * np.hstack([x_centers, y_centers])

    list_covariance_matrices = []
    for vp, angle, i in zip(
        cycle(list_concentrations), cycle(list_angles), range(n_blobs)
    ):
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        # rotate the covariance matrix
        cov_matrix = rotation_matrix @ np.diag(vp) @ rotation_matrix.T

        list_covariance_matrices.append(cov_matrix)

    # n samples per blobs
    n_per_blobs = rng.multinomial(n_samples, np.ones(n_blobs) / n_blobs)

    list_gmm_samples = []

    for i in range(n_blobs):
        n_current_gaussian = n_per_blobs[i]
        gaussian_samples = rng.multivariate_normal(
            means[i], list_covariance_matrices[i], n_current_gaussian
        )
        list_gmm_samples.append(gaussian_samples)

    samples = np.vstack(list_gmm_samples)

    if uniform_noise is not None:
        outliers = rng.uniform(
            low=-d * (max_d + 2), high=d * (max_d + 2), size=(uniform_noise, 2)
        )
        samples = np.vstack((samples, outliers))

    if return_parameter:
        return samples, means, list_covariance_matrices

    return samples
