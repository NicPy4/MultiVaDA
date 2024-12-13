from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import umap


# dataset = fetch_openml('Fashion-MNIST')

dataset = fetch_openml('mnist_784')

# dataset = fetch_openml('iris')
# dataset.target = dataset.target.map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})


# Standard UMAP clustering
for dim in [2, 3]:
    for n_neighbors in [15]:
        for min_dist in [0.0, 0.1, 1.0]:
            for metric in ["euclidean", "correlation"]:
                standard_embedding = (umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=dim,
                    metric=metric)
                                      .fit_transform(dataset.data))
                if dim == 1:
                    plt.scatter(standard_embedding[:, 0],
                                range(len(standard_embedding)),
                                c=dataset.target.astype(int),
                                s=0.1,
                                cmap='Spectral')
                elif dim == 2:
                    plt.scatter(standard_embedding[:, 0],
                                standard_embedding[:, 1],
                                c=dataset.target.astype(int),
                                s=0.1,
                                cmap='Spectral')
                elif dim == 3:
                    ax = plt.figure().add_subplot(projection="3d")
                    ax.scatter(standard_embedding[:, 0],
                               standard_embedding[:, 1],
                               standard_embedding[:, 2],
                               c=dataset.target.astype(int),
                               s=0.1,
                               cmap='Spectral')
                plt.title(f"UMAP: {dim}-D, n_neighbors = {n_neighbors}, min_dist = {min_dist},\nmetric = {metric}")
                plt.savefig(f"fashion_plots/umap_{dim}_{n_neighbors}_{min_dist}_{metric}.pdf")
                plt.show()
