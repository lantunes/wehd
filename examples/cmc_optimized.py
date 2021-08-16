from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from wehd import WEHD, cluster_variance
import numpy as np
from gradient_free_optimizers import EvolutionStrategyOptimizer
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    n_clusters = 3

    X_orig = []
    y = []
    with open("../data/uci_contraceptive_dataset/cmc.data", "rt") as f:
        """
        1. Wife's age                   (numerical)
        2. Wife's education             (numerical, 1=low, 2, 3, 4=high)
        3. Husband's education          (numerical, 1=low, 2, 3, 4=high)
        4. Number of children ever born (numerical)
        5. Wife's religion              (categorical, 0=Non-Islam, 1=Islam)
        6. Wife's now working?          (categorical, 0=Yes, 1=No)
        7. Husband's occupation         (categorical, 1, 2, 3, 4)
        8. Standard-of-living index     (numerical, 1=low, 2, 3, 4=high)
        9. Media exposure               (categorical, 0=Good, 1=Not good)
        """
        for line in f.readlines():
            line = line.strip()
            values = [float(v) for v in line.split(",")]
            X_orig.append(values[:9])
            y.append(int(values[-1]) - 1)
    X = np.array(X_orig)
    y = np.array(y)
    numeric_features = [0, 1, 2, 3, 7]
    categorical_features = [4, 5, 6, 8]

    scaler = MinMaxScaler().fit(X[:, numeric_features])
    X[:, numeric_features] = scaler.transform(X[:, numeric_features])


    def model(para):
        weights = np.array([para["w%s" % (i + 1)] for i in range(9)])
        tot = np.sum(weights)
        if tot > 0:
            weights = weights / tot
        w = WEHD(categorical_indices=categorical_features, weights=weights)

        D = w.get_distance_matrix(X)

        kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed").fit(D)
        labels = kmedoids.labels_

        clusters = set(labels)

        SSC = 0  # the within cluster variance
        for label in clusters:
            X_indices = [i for i, v in enumerate(labels) if v == label]
            SSC += cluster_variance(X_indices, D)

        SST = cluster_variance(list(range(len(X))), D)  # the total variance

        return -SSC/SST  # we want to minimize SSC/SST


    search_space = {"w%s" % (i + 1): np.linspace(0, 21, 22) for i in range(9)}

    opt = EvolutionStrategyOptimizer(search_space)
    opt.search(model, n_iter=500)

    best_params = opt.best_para
    weights = np.array([best_params["w%s" % (i + 1)] for i in range(9)])
    w = WEHD(categorical_indices=categorical_features, weights=weights / np.sum(weights))
    D = w.get_distance_matrix(X)

    kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed").fit(D)
    labels = kmedoids.labels_

    print("Rand Index of K-medoids WEHD-optimized classifier: %s" % rand_score(y, labels))

    # re-map the cluster labels so that they match class labels
    labels_remapped = np.zeros_like(labels)
    for i in range(n_clusters):
        mask = (labels == i)
        labels_remapped[mask] = mode(y[mask])[0]
    print("accuracy score: %s" % accuracy_score(y, labels_remapped))
    print("confusion matrix: \n%s" % confusion_matrix(y, labels_remapped))

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, learning_rate=10, metric="precomputed")
    result = tsne.fit_transform(D)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_clusters-1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Accent)
    colors = [mapper.to_rgba(label) for label in labels]

    fig = plt.figure()
    plt.scatter(result[:, 0], result[:, 1], c=colors, marker="o", edgecolors="black", picker=True)

    def onpick(event):
        ind = event.ind
        print()
        for i in ind:
            print("features: %s, true label: %s" % (X_orig[i], y[i]))

    fig.canvas.mpl_connect("pick_event", onpick)

    plt.show()
