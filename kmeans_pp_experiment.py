import argparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.neighbors import NearestNeighbors

from dataset import gaussian_blobs
from dataset_experiment import dataset_parser
from experiments import exponent_experiment, get_n_outliers
from kmeans import KMeans, kmeans_pp_init, phi, random_init
from visualisation import save_fig

parser = argparse.ArgumentParser(
    description=(
        "This program executes a series of 2 experiments on KMeans++ properties."
        "First one simple compares visually KMeans++ with a random initialisation."
        "Second one compares the quality of the initialisation of KMeans++ based on the value of the exponent."
        "For the second experiment, we look at the number of selected outliers and the average result of the potential function after 100 iterations of KMeans algorithm."
    ),
    parents=[dataset_parser],
)
args = parser.parse_args()

print("KMeans pp investigation. Use --show to print the figures.")
print("Experiment 1 - KMeans vs random initialisation...")
n_uniform = int(args.n_samples * args.ratio_uniform)
samples = gaussian_blobs(
    d=args.distance,
    uniform_noise=n_uniform,
    n_blobs=args.n_clusters,
    n_samples=args.n_samples,
)
kmeanspp_centers = kmeans_pp_init(samples, args.n_clusters, seed=5, p=2)
random_centers = random_init(samples, args.n_clusters, seed=10)

fig = make_subplots(
    rows=1, cols=2, subplot_titles=["kmeans++", "random initialization"]
)
labels = (1 + np.arange(kmeanspp_centers.shape[0])).astype(str)
text = [f"<b>{label}</b>" for label in labels]
trace_X = go.Scatter(
    x=samples[:, 0],
    y=samples[:, 1],
    mode="markers",
    marker=dict(color="#a5b3cf"),
    showlegend=False,
)
fig.add_traces([trace_X, trace_X], rows=1, cols=[1, 2])

fig.add_traces(
    px.scatter(
        x=kmeanspp_centers[:, 0], y=kmeanspp_centers[:, 1], color=labels, text=text
    ).data,
    rows=1,
    cols=1,
)
fig.add_traces(
    px.scatter(
        x=random_centers[:, 0], y=random_centers[:, 1], color=labels, text=text
    ).data,
    rows=1,
    cols=2,
)

fig.update_layout(
    title="Initialization influence",
    margin={"t": 50, "r": 5, "l": 5, "b": 5},
    width=700,
    height=300,
    showlegend=False,
)
fig.update_yaxes(scaleanchor="x1", scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
fig.update_traces(textposition="top center")
fig.update_annotations(font=dict())

save_fig(fig, "kmeans-random-init.pdf", args.save_path, show=args.show)

print("Experiment 2 - Studying the influence of p...")
n_rows = 3
n_cols = 6
list_p = [0, 1, 2, 4, 6, 15]

subplot_titles = []

for i in range(n_rows):
    for j in range(n_cols):
        subplot_titles.append(f"p={list_p[j]}")
fig = make_subplots(
    rows=n_rows, cols=n_cols, vertical_spacing=0.075, subplot_titles=subplot_titles
)
labels = (1 + np.arange(kmeanspp_centers.shape[0])).astype(str)
text = [f"<b>{label}</b>" for label in labels]

trace_X = go.Scatter(
    x=samples[:, 0],
    y=samples[:, 1],
    mode="markers",
    marker=dict(color="#a5b3cf"),
    showlegend=False,
)
n_plot = 0
for i in range(1, n_rows + 1):
    for j in range(1, n_cols + 1):
        if j == 1:
            centers = random_init(
                samples,
                args.n_clusters,
                seed=i,
            )
        else:
            centers = kmeans_pp_init(samples, args.n_clusters, seed=i, p=list_p[j - 1])
        fig.add_trace(trace_X, row=i, col=j)
        fig.add_traces(
            px.scatter(x=centers[:, 0], y=centers[:, 1], color=labels, text=text).data,
            rows=i,
            cols=j,
        )
        fig.layout.annotations[n_plot].update(text=f"p={list_p[j-1]}")

        n_plot += 1
        fig.update_yaxes(scaleanchor=f"x{n_plot}", scaleratio=1, row=i, col=j)

fig.update_layout(
    title="Exponent influence, 50 experiments",
    margin={"t": 50, "r": 5, "l": 5, "b": 5},
    width=900,
    height=700,
    showlegend=False,
)

fig.update_traces(textposition="top center")
fig.update_annotations(font=dict())
save_fig(fig, "p-on-data.pdf", args.save_path, args.show)

n_experiments = 50
list_p = np.array([0, 1, 2, 3, 4, 5, 6, 7])
p_outliers = exponent_experiment(
    list_p,
    n_experiments,
    "outliers",
    d=args.distance,
    ratio_uniform=args.ratio_uniform,
    rotation="random",
    n_samples=args.n_samples,
)

n_experiments = 50
p_phi = exponent_experiment(
    list_p,
    n_experiments,
    "kmeans+phi",
    d=args.distance,
    ratio_uniform=args.ratio_uniform,
    rotation="random",
    n_samples=args.n_samples,
)

fig = go.Figure()


for i, p in enumerate(list_p):
    fig.add_trace(
        go.Violin(
            x=[p for _ in range(n_experiments)],
            y=p_outliers[i],
            name=f"{p}",
            box_visible=True,
            meanline_visible=True,
        )
    )
fig.update_layout(
    title="Average number of selected outliers, 50 experiments",
    margin={"t": 50, "r": 5, "l": 5, "b": 5},
    width=600,
    height=300,
    showlegend=True,
    legend_title_text="p",
)
fig.update_yaxes(title="Average number of outliers")
fig.update_xaxes(title="p", tickmode="linear")
save_fig(fig, "p-analysis-violin.pdf", args.save_path, show=args.show)


fig = px.bar(x=list_p, y=p_outliers.mean(axis=1), color=list_p.astype(str))

fig.update_layout(
    title="Repartition of the number of selected outliers, 50 experiments",
    margin={"t": 50, "r": 5, "l": 5, "b": 5},
    width=600,
    height=300,
    showlegend=True,
    legend_title_text="p",
)
fig.update_yaxes(title="Average number of outliers")
fig.update_xaxes(title="p", tickmode="linear")

save_fig(fig, "p-analysis-bar.pdf", args.save_path, show=args.show)

fig = px.bar(x=list_p, y=p_phi.mean(axis=1), color=list_p.astype(str))

fig.update_layout(
    title="Exponent influence on phi, 50 experiments",
    margin={"t": 50, "r": 5, "l": 5, "b": 5},
    width=600,
    height=300,
    showlegend=True,
    legend_title_text="p",
)
fig.update_yaxes(title="Average phi value")
fig.update_xaxes(title="p", tickmode="linear")
save_fig(fig, "phi-analysis-bar.pdf", args.save_path, show=args.show)


fig = go.Figure()


for i, p in enumerate(list_p):
    fig.add_trace(
        go.Violin(
            x=[p for _ in range(n_experiments)],
            y=p_phi[i],
            name=f"{p}",
            box_visible=True,
            meanline_visible=True,
        )
    )
fig.update_layout(
    title="Exponent influence on phi, 50 experiments",
    margin={"t": 50, "r": 5, "l": 5, "b": 5},
    width=600,
    height=300,
    showlegend=True,
    legend_title_text="p",
)
fig.update_yaxes(title="Average phi value")
fig.update_xaxes(title="p", tickmode="linear")

save_fig(fig, "phi-analysis-violin.pdf", args.save_path, show=args.show)


def exponent_experiment_report(list_metric, list_p):
    list_dict = []
    for i in range(len(list_p)):
        dict_exp = {
            "p": list_p[i],
            "mean": list_metric[i].mean(),
            "std": list_metric[i].std(),
            "min": list_metric[i].min(),
            "max": list_metric[i].max(),
        }
        list_dict.append(dict_exp)
    df_experiment = pd.DataFrame(list_dict)
    df_experiment = df_experiment.set_index("p", drop=True)
    with pd.option_context("display.precision", 3):
        print(df_experiment)


print("p influence on phi on 50 experiment:")
exponent_experiment_report(p_phi, list_p)
