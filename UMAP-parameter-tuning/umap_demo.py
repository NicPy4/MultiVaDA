import umap
import numpy as np
import polars as pl
import matplotlib.pyplot as plt


# Define plotter function
def draw_umap(data_set, n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", title=""):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data_set)

    if n_components == 1:
        plt.scatter(u[:, 0], range(len(u)))
    if n_components == 2:
        plt.scatter(u[:, 0], u[:, 1])
    if n_components == 3:
        ax = plt.figure().add_subplot(projection="3d")
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=100)
    plt.title(title, fontsize=18)


# Read in the two csv data files
def read_heave_data(path: str) -> pl.DataFrame:
    return pl.read_csv(path).with_columns(pl.col("time").str.to_datetime("%m/%d/%y %I:%M %p"),
                                          pl.col("ship").str.split(" ").list.last().str.to_integer(base=10),
                                          pl.col("type").str.replace("cargo", "Cargo").str.replace("General Cargo", "0").str.replace("Bulkcarrier", "1").cast(pl.Int64))


test_data = read_heave_data("heave_test.csv")
train_data = read_heave_data("heave_train.csv")

# Describe the test data
print(train_data)
print(train_data.describe())

# Reduce the dimensionality of the data to 1, 2 and 3 dimensions and visualize the results
for neighbours in range(5, 46, 10):
    for distance in np.arange(0, 0.5, 0.1):
        for metric in ["euclidean", "chebyshev", "correlation"]:
            for dim in range(2, 4):
                draw_umap(train_data,
                          n_neighbors=neighbours,
                          min_dist=distance,
                          n_components=dim,
                          metric=metric,
                          title=f"Neighbours = {neighbours}, Distance = {distance:.1f},\nMetric = {metric}")
                plt.savefig(f"ship_plots/umap_{neighbours}_{distance:.1f}_{dim}_{metric}.pdf")
                plt.show()
