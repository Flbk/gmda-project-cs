# gmda-project-cs

## Installation
Simply build a virtual environment using `virtualenv` or `conda` and install the requirements.

```bash
virtualenv env-nmf-clustering
source env-nmf-clustering/bin/activate
pip install -r requirements.txt
```

## Project organization
To avoid any importation error we have a flat organization (all the files are in the same main folder).

Here are the description of each file.

### Classes and function
`dataset.py`: functions to build the parametric dataset.

`kmeans.py`: KMeans class and related functions like KMeans++.

`nmf.py`: NMF classes.

`visualization.py`: Utility function to plot our figures using `plotly`.

### Scripts
All the outputs presented in the report are from Python scripts found here. They can all be launched using Python CLI. Lauchning them with `--help` will print a description of the different option. But they can be launched as it, with coherent default.

If you want the figures to be printed on screen, use the `--show` arguent (not default).

`dataset_experiment.py`: A small script to play with the parametric dataset.

`kmeans_pp_experiment.py`: The experiments related to KMeans++ initialization.

`nmf_experiments.py`: The experiments related to 2-NMF and 3-NMF.

`data_embedding.py`: A small experiments in which we embed the 2D data into a high dimensional space.

### Report and articles
The cited articles are in the folder `articles`. The project report (mandatory deliverable for the course) is `project_report.pdf`.
