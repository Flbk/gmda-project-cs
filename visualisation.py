import plotly.io as pio
from pathlib import Path


def save_fig(fig, file_name, dir_path, show=False):
    if show:
        fig.show()
    if file_name != "":
        save_file = Path(dir_path) / file_name
        save_file.parent.mkdir(exist_ok=True, parents=True)
        pio.write_image(fig, save_file)
        print(f"Figure saved at: {save_file}")
