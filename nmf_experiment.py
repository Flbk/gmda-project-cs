import argparse
import time

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
    samples = gaussian_blobs(
        d=1.5, uniform_noise=n_uniform, n_samples=args.n_samples, non_negative=True
    )

    # Fitting NMF with random init
    nmf = NMF(
        n_clusters=args.n_clusters,
        init="random",
        max_iter=args.nmf_iter,
        tolerance=-np.inf,
        seed=args.clustering_seed,
    )

    F, G, labels = nmf.fit_predict(samples)

    ## Plotting the orthogonality of G.
    ratio_points = 1 / args.n_clusters
    nmf_centers_scaled = F.T * np.quantile(G, 1 - ratio_points, axis=0)[:, None]
    nmf_centers = F.T

    G_dot = G.T @ G
    fig = px.imshow(G_dot)
    fig.update_layout(margin={"t": 5, "b": 5, "l": 5, "r": 5}, width=300, height=300)
    fig.update_xaxes(tickmode="linear")
    save_fig(fig, "GGT_nmf.pdf", args.save_path, show=args.show)

    # Needed to avoid a display bug.
    time.sleep(2)
    save_fig(fig, "GGT_nmf.pdf", args.save_path, show=args.show)

    # Fitting NMF with Kmeans++ init
    nmf = NMF(
        n_clusters=args.n_clusters,
        init="kmeanspp",
        max_iter=args.nmf_iter,
        tolerance=-np.inf,
        seed=args.clustering_seed,
    )
    F_pp, G_pp, labels_pp = nmf.fit_predict(samples)
    nmf_kmeanspp_centers = F_pp.T
    nmf_kmeanspp_centers_scaled = (
        F_pp.T * np.quantile(G_pp, 1 - ratio_points, axis=0)[:, None]
    )

    # Fitting basic KMeans
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        seed=args.clustering_seed,
    )
    kmeans_centers, kmeans_labels = kmeans.fit_predict(samples)

    # Plotting NMF without scaling
    fig = plot_several_centers(
        samples,
        [kmeans_centers, nmf_centers, nmf_kmeanspp_centers],
        list_names=["kmeans++", "nmf random", "nmf kmeans++"],
        list_colors=cmap,
        title="Kmeans++ and 2-NMF",
    )
    save_fig(fig, "nmf-kmeans-centers.pdf", args.save_path, show=args.show)

    # Plotting NMF with scaling
    fig = plot_several_centers(
        samples,
        [kmeans_centers, nmf_centers_scaled, nmf_kmeanspp_centers_scaled],
        list_names=["kmeans++", "nmf random", "nmf kmeans++"],
        list_colors=cmap,
        title="Kmeans++ and 2-NMF",
    )
    save_fig(fig, "nmf-kmeans-centers-scaled.pdf", args.save_path, show=args.show)

    # Plotting random NMF clustering
    fig = plot_clustering(samples, labels=labels)
    save_fig(fig, "nmf-clustering.pdf", args.save_path, show=args.show)

    # Plotting KMeans++ initialized NMF clustering
    fig = plot_clustering(samples, labels=labels_pp)
    save_fig(fig, "nmf-kmeanspp-clustering.pdf", args.save_path, show=args.show)

    # Experiment 2 - 3-NMF
    print("Experiment 2 - TriNMF")

    # Fitting 3 NMF
    tri_nmf = TriNMF(
        n_clusters=args.n_clusters,
        init="kmeanspp",
        max_iter=args.nmf_iter,
        tolerance=-np.inf,
        seed=args.clustering_seed,
    )
    F_tri, S_tri, G_tri, labels_tri = tri_nmf.fit_predict(samples)

    # Plotting reconstruction
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=tri_nmf.list_errors,
            x=np.arange(len(tri_nmf.list_errors)),
            name="reconstruction error",
        )
    )
    fig.update_layout(
        title=None,
        margin={"t": 5, "r": 5, "l": 5, "b": 5},
        showlegend=True,
        width=400,
        height=300,
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    )
    save_fig(fig, "loss-s-diag.pdf", args.save_path, show=args.show)
    reconstructed_X = F_tri @ S_tri @ G_tri.T

    fig = plot_centers(samples, reconstructed_X.T, name="reconstruced", width=400)
    fig.update_layout(
        title=None,
        margin={"t": 5, "r": 5, "l": 5, "b": 5},
        showlegend=True,
        width=400,
        height=300,
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    )
    save_fig(fig, "reconstructed-3-nmf.pdf", args.save_path, show=args.show)

    # Plotting orthogonality of F and G
    ratio_points = 1 / args.n_clusters
    tri_nmf_centers_scaled = (
        F_tri.T * np.quantile(S_tri @ G_tri.T, 1 - ratio_points, axis=1)[:, None]
    )
    tri_nmf_centers = F_tri.T

    G_tri_dot = G_tri.T @ G_tri
    F_tri_dot = F_tri.T @ F_tri
    fig = px.imshow(G_tri_dot)
    fig.update_layout(margin={"t": 5, "b": 5, "l": 5, "r": 5}, width=300, height=300)
    fig.update_xaxes(tickmode="linear")
    save_fig(fig, "GGT_tri_nmf.pdf", args.save_path, show=args.show)

    fig = px.imshow(F_tri_dot)
    fig.update_layout(margin={"t": 5, "b": 5, "l": 5, "r": 5}, width=300, height=300)
    fig.update_xaxes(tickmode="linear")
    save_fig(fig, "FFT_tri_nmf.pdf", args.save_path, show=args.show)

    # Plotting 3-NMF without rescaling.
    fig = plot_several_centers(
        samples,
        [kmeans_centers, tri_nmf_centers, nmf_kmeanspp_centers],
        list_colors=cmap,
        list_names=["kmeans++", "3-NMF", "2-NMF"],
        title="KMeans++ and 3-NMF",
    )
    save_fig(fig, "2-3-nmf.pdf", args.save_path, show=args.show)

    # Plotting 3-NMF with rescaling
    fig = plot_several_centers(
        samples,
        [kmeans_centers, tri_nmf_centers_scaled, nmf_kmeanspp_centers_scaled],
        list_colors=cmap,
        list_names=["kmeans++", "3-NMF", "2-NMF"],
        title="KMeans++ and NMF",
    )
    save_fig(fig, "2-3-nmf-scaled.pdf", args.save_path, show=args.show)

    # Plotting 3-NMF clustering
    fig = plot_clustering(samples, labels=labels_tri)
    save_fig(fig, "3-nmf-clustering.pdf", args.save_path, show=args.show)
