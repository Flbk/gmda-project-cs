from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None
cmap = px.colors.qualitative.Plotly


def save_fig(fig, file_name, dir_path, show=False):
    if show:
        fig.show()
    if file_name != "":
        save_file = Path(dir_path) / file_name
        save_file.parent.mkdir(exist_ok=True, parents=True)
        pio.full_figure_for_development(fig, warn=False)
        pio.write_image(fig, save_file)
        print(f"Figure saved at: {save_file}")


def plot_centers(
    samples, centers, name="", color="red", title="", width=700, height=300
):
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
    samples, list_centers, list_names, list_colors, width=400, height=300, title=None
):
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


def plot_clustering(X, labels, width=400, height=300):
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
