from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None
cmap = px.colors.qualitative.Plotly


def save_fig(fig: go.Figure, file_name: str, dir_path: str, show: bool = False):
    """Save and print plotly figures.

    Args:
        fig (go.Figure): The plotly figure to save.
        file_name (str): File name.
        dir_path (str): Directory where to save the file.
        show (bool, optional): Whether to show the figure. Defaults to False.
    """
    if show:
        fig.show()
    if dir_path != None:
        save_file = Path(dir_path) / file_name
        save_file.parent.mkdir(exist_ok=True, parents=True)
        pio.full_figure_for_development(fig, warn=False)
        pio.write_image(fig, save_file)
        print(f"Figure saved at: {save_file}")


def plot_centers(
    samples: np.ndarray,
    centers: np.ndarray,
    name: str = "",
    color: str = "red",
    title: str = "",
    width: int = 700,
    height: int = 300,
):
    """Plot a dataset of 2D points and highlight some points.
    In most cases these highlighted points will be interpreted as cluster's centers.

    Args:
        samples (np.ndarray, (N, 2)): The 2D dataset.
        centers (np.ndarray, (K, 2)): M 2D cluster's centers.
        name (str, optional): The name of the dataset for the legend. Defaults to "".
        color (str, optional): The color for the centers. Defaults to "red".
        title (str, optional): The title of the figure. Defaults to "".
        width (int, optional): Width in px. Defaults to 700.
        height (int, optional): Height in px. Defaults to 300.

    Returns:
        go.Figure: The plotly figure.
    """
    fig = go.Figure()
    trace_X = go.Scatter(
        x=samples[:, 0],
        y=samples[:, 1],
        mode="markers",
        marker=dict(color="#a5b3cf"),
        showlegend=False,
    )
    fig.add_trace(trace_X)

    fig.add_traces(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker=dict(color=cmap[0]),
            name=name,
        ),
    )
    fig.update_layout(
        title=title,
        margin={"t": 50, "r": 5, "l": 5, "b": 5},
        width=width,
        height=height,
        showlegend=True,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_traces(textposition="top center")
    fig.update_annotations(font=dict())
    return fig


def plot_several_centers(
    samples: np.ndarray,
    list_centers: Iterable[np.ndarray],
    list_names: Iterable[str],
    list_colors: Iterable[str],
    width: int = 400,
    height: int = 300,
    title: str = None,
):
    """Plot several highlighted batch of centers on top of a 2D dataset.

    Args:
        samples (np.ndarray, (N, 2)): The dataset samples.
        list_centers (Iterable[np.ndarray]): List of (K, 2) arrays containing centers from different ouputs.
        list_names (Iterable[str]): List of legend names of each batch of centers.
        list_colors (Iterable[str]): The colors for each batch of center.
        width (int, optional): Figure's width. Defaults to 400.
        height (int, optional): Figure's height. Defaults to 300.
        title (str, optional): Figure's title. Defaults to None.

    Returns:
        go.Figure: The plotly figure.
    """
    fig = plot_centers(
        samples,
        list_centers[0],
        list_names[0],
        list_colors[0],
        width=width,
        height=height,
        title=title,
    )
    for i in range(1, len(list_centers)):
        fig.add_traces(
            go.Scatter(
                x=list_centers[i][:, 0],
                y=list_centers[i][:, 1],
                mode="markers",
                marker=dict(color=list_colors[i]),
                name=list_names[i],
            ),
        )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    )
    return fig


def plot_clustering(
    X: np.ndarray, labels: np.ndarray, width: int = 400, height: int = 300
):
    """Plot a clustering (i.e each point in the same cluster will have the same color).

    Args:
        X (np.ndarray, (N, 2)): The 2D dataset.
        labels (np.ndarray, (N,)): The labels of each point.
        width (int, optional): The width of the figure. Defaults to 400.
        height (int, optional): The height of the figure. Defaults to 300.

    Returns:
        go.Figure: The plotly figure.
    """
    ind_sort = np.argsort(labels)
    X = X[ind_sort]
    labels = labels[ind_sort]
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels.astype(str))
    fig.update_layout(
        margin={"t": 5, "r": 5, "l": 5, "b": 5},
        width=width,
        height=height,
        showlegend=True,
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title=None)
    fig.update_xaxes(title=None)
    fig.update_layout(legend_title_text=None)
    return fig
