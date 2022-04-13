import argparse

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dataset import gaussian_blobs
from dataset_experiment import dataset_parser
from kmeans import KMeans
from nmf import NMF, TriNMF
from visualisation import plot_centers, plot_clustering, plot_several_centers, save_fig

parser = argparse.ArgumentParser(parents=[dataset_parser])
parser.add_argument("--nmf_init", help="NMF initialization.", default="random")
parser.add_argument(
    "--nmf_iter", help="Number of iterations for NMF.", type=int, default=3000
)

parser.add_argument(
    "--clustering_seed", help="Seed for the NMF and KMeans.", type=int, default=0
)
if __name__ == "__main__":

    args = parser.parse_args()

    cmap = px.colors.qualitative.Plotly
    print("Experiment 1 - Comparing NMF ad KMeans output.")
    n_uniform = int(args.n_samples * args.ratio_uniform)
    samples_2d = gaussian_blobs(
        d=1.5, uniform_noise=n_uniform, n_samples=args.n_samples, non_negative=True
    )

    # Embedd the samples
    samples = np.zeros((samples_2d.shape[0], 50))
    samples[:, :2] = samples_2d

    # Fitting 2-NMF
    nmf = NMF(
        n_clusters=args.n_clusters,
        init="kmeanspp",
        max_iter=args.nmf_iter,
        tolerance=-np.inf,
        seed=args.clustering_seed,
    )
    F, G, labels_nmf = nmf.fit_predict(samples)

    # Fitting 3-NMF
    tri_nmf = TriNMF(
        n_clusters=args.n_clusters,
        init="kmeanspp",
        max_iter=args.nmf_iter,
        tolerance=-np.inf,
        seed=args.clustering_seed,
    )
    F_tri, S_tri, G_tri, labels_tri = tri_nmf.fit_predict(samples)

    # Fitting KMeans
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        seed=args.clustering_seed,
    )
    kmeans_centers, kmeans_labels = kmeans.fit_predict(samples)

    # Plotting 2-NMF clustering
    fig = plot_clustering(samples_2d, labels=labels_nmf)
    save_fig(fig, "2-nmf-embedded.pdf", args.save_path, show=args.show)
    # Plotting 3-NMF clustering
    fig = plot_clustering(samples_2d, labels=labels_tri)
    save_fig(fig, "3-nmf-embedded.pdf", args.save_path, show=args.show)
    # Plotting KMeans clustering
    fig = plot_clustering(samples_2d, labels=kmeans_labels)
    save_fig(fig, "kmeans-embedded.pdf", args.save_path, show=args.show)
