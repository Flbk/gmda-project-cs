from typing import Tuple

import numpy as np

from kmeans import kmeans_pp_init


class NMF:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 5000,
        tolerance: float = 1e-7,
        seed: int = 0,
        init: str = "random",
    ):
        """2-NMF $X = FG^T$ with orthogonality constraints on G.

        Args:
            n_clusters (int): The number of dictionary atoms.
            max_iter (int, optional): The maximum number of iterations of the algorithm. Defaults to 5000.
            tolerance (float, optional): The minimum relative error decrease to reach before stopping the iterations. Defaults to 1e-7.
            seed (int, optional): Random seed. Defaults to 0.
            init (str, optional): Which initialization to use. One of "random", "kmeans++". Defaults to "random".
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.seed = seed
        self.init = init

    def fit(self, X: np.ndarray):
        """Fit the NMF with orthogonality constraints.

        Args:
            X (np.ndarray): The samples on which to fit the NMF.

        Returns:
            Tuple[np.ndarray, np.ndarray]: F and G factor matrices of X.
        """
        # Transpose to get the classical NMF problem.
        X = X.T
        p, N = X.shape
        rng = np.random.default_rng(seed=self.seed)
        if self.init == "random":
            F = rng.uniform(size=(p, self.n_clusters))
            G = rng.uniform(size=(N, self.n_clusters))

        if self.init == "kmeanspp":
            print("kmeanspp")
            F = kmeans_pp_init(X.T, self.n_clusters, seed=self.seed).T
            G = rng.uniform(size=(N, self.n_clusters))

        list_errors = []
        convergence = False
        n_iter = 0
        while not convergence:
            G = G * (np.dot(F.T, X) / (np.linalg.multi_dot((G.T, X.T, F, G.T)))).T
            F = F * np.dot(X, G) / np.linalg.multi_dot((F, G.T, G))

            error = ((X - F @ G.T) ** 2).mean()
            list_errors.append(error)
            n_iter += 1
            if n_iter > 1:
                prev_error = list_errors[-2]
                rel_error_decrease = np.abs(prev_error - error) / prev_error
                convergence = (rel_error_decrease < self.tolerance) or (
                    n_iter >= self.max_iter
                )
        self.F = F
        self.G = G
        self.list_errors = list_errors
        return self

    def fit_predict(self, X: np.ndarray):
        """A wrapper to fit and predict samples.

        Args:
            X (np.ndarray): The samples on which to fit the NMF.

        Returns:
            Tuple[np.ndarray, np.ndarray]: F and G factor matrices of X.
        """
        self.fit(X)
        labels = self.G.argmax(axis=1)
        return self.F, self.G, labels


class TriNMF:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 5000,
        tolerance: float = 1e-7,
        diagonal: bool = True,
        seed: int = 0,
        init: str = "random",
    ):
        """3-NMF $X = F S G^T$, with orthogonality constraints on F and G.

        Args:
            n_clusters (int): The number of dictionary atoms.
            max_iter (int, optional): The maximum number of iterations of the algorithm. Defaults to 5000.
            tolerance (float, optional): The minimum relative error decrease to reach before stopping the iterations. Defaults to 1e-7.
            diagonal (bool, optional): _description_. Defaults to True.
            seed (int, optional): Random seed. Defaults to 0.
            init (str, optional): Which initialization to use. One of "random", "kmeans++". Defaults to "random".
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.seed = seed
        self.init = init
        self.diagonal = diagonal

    def fit(self, X: np.ndarray):

        """Fit the 3-NMF with orthogonality constraints.

        Args:
            X (np.ndarray): The samples on which to fit the NMF.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: F, S and G factor matrices of X.
        """
        # Transpose to get the classical NMF problem.
        X = X.T
        p, N = X.shape
        rng = np.random.default_rng(seed=self.seed)
        if self.init == "random":
            F = rng.uniform(size=(p, self.n_clusters))
        if self.init == "centered":
            F = np.zeros((p, self.n_clusters))
            for i in range(p):
                min_val, max_val = X[i, :].min(), X[i, :].max()
                F[i, :] = rng.uniform(min_val, max_val, size=(self.n_clusters))

        if self.init == "kmeanspp":
            print("kmeanspp")
            F = kmeans_pp_init(X.T, self.n_clusters, seed=self.seed).T

        G = rng.uniform(size=(N, self.n_clusters))
        if self.diagonal:
            S = rng.uniform(size=(self.n_clusters))
            S = np.diag(S)
        else:
            S = rng.uniform(size=(self.n_clusters, self.n_clusters))

        list_errors = []

        convergence = False
        n_iter = 0
        while not convergence:

            G = G * np.sqrt(
                np.linalg.multi_dot((X.T, F, S))
                / np.linalg.multi_dot((G, G.T, X.T, F, S))
            )
            F = F * np.sqrt(
                np.linalg.multi_dot((X, G, S.T))
                / np.linalg.multi_dot((F, F.T, X, G, S.T))
            )
            S = S * np.sqrt(
                np.linalg.multi_dot((F.T, X, G))
                / np.linalg.multi_dot((F.T, F, S, G.T, G))
            )

            error = ((X - F @ S @ G.T) ** 2).mean()
            list_errors.append(error)
            n_iter += 1
            if n_iter > 1:
                prev_error = list_errors[-2]
                rel_error_decrease = np.abs(prev_error - error) / prev_error
                convergence = (rel_error_decrease < self.tolerance) or (
                    n_iter >= self.max_iter
                )
        self.F = F
        self.G = G
        self.S = S
        self.list_errors = list_errors

    def fit_predict(self, X: np.ndarray):
        """A wrapper to fit and predict samples.

        Args:
            X (np.ndarray): The samples on which to fit the NMF.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: F, S and G factor matrices of X.
        """
        self.fit(X)
        labels = self.G.argmax(axis=1)
        return self.F, self.S, self.G, labels
