# %%
import argparse
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dataset import gaussian_blobs


class ConvertDegreesAction(argparse.Action):
    def __call__(self, parser, namespace, values, **kwargs):
        setattr(namespace, self.dest, np.deg2rad(values))


dataset_parser = argparse.ArgumentParser(
    description="Generate a parametrized dataset with mixture of gaussians.",
    add_help=False,
)

dataset_parser.add_argument(
    "-c", "--n_clusters", help="Number of clusters.", type=int, default=5
)

dataset_parser.add_argument(
    "-d",
    "--distance",
    help="Relative distance between the centers.",
    type=float,
    default=1.5,
)
dataset_parser.add_argument(
    "-n",
    "--n_samples",
    help="The number of samples to draw on the dataset.",
    type=int,
    default=500,
)
dataset_parser.add_argument(
    "-r",
    "--ratio_uniform",
    default=0.1,
    type=float,
    help="The percentage of uniform noise on the dataset.",
)
dataset_parser.add_argument(
    "--rotation",
    help="The rotation (in degrees) for each cluster.",
    type=float,
    nargs="+",
    default=[0, 45, 0, 90, -45],
)
dataset_parser.add_argument(
    "--save_path",
    help="Where to save the figures. If empty the figures won't be saved.",
    default="",
)
dataset_parser.add_argument(
    "--show",
    help="Whether to print the figures during execution.",
    action="store_true",
)
# %%

# %%
if __name__ == "__main__":
    args = dataset_parser.parse_args()
    samples = gaussian_blobs(d=2, max_d=3, n_blobs=args.n_clusters)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples[:, 0], y=samples[:, 1], mode="markers"))
    fig.update_layout(margin={"t": 5, "r": 5, "l": 5, "b": 5}, width=500, height=300)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

# %%
