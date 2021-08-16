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
    digits = datasets.load_digits()

    X = digits.data
    y = digits.target

    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    def model(para):
        weights = np.array([para["w%s" % (i+1)] for i in range(64)])
        tot = np.sum(weights)
        if tot > 0:
            weights = weights / tot
        w = WEHD(categorical_indices=[], weights=weights)

        D = w.get_distance_matrix(X)

        kmedoids = KMedoids(n_clusters=10, metric="precomputed").fit(D)
        labels = kmedoids.labels_

        clusters = set(labels)

        SSC = 0  # the within cluster variance
        for label in clusters:
            X_indices = [i for i, v in enumerate(labels) if v == label]
            SSC += cluster_variance(X_indices, D)

        SST = cluster_variance(list(range(len(X))), D)  # the total variance

        return -(SSC / SST)  # we want to minimize SSC/SST

    search_space = {"w%s" % (i+1): np.linspace(0, 21, 22) for i in range(64)}

    opt = EvolutionStrategyOptimizer(search_space, initialize={"random": 10})
    opt.search(model, n_iter=200)

    best_params = opt.best_para
    weights = [best_params["w%s" % (i+1)] for i in range(64)]
    w = WEHD(categorical_indices=[], weights=weights)
    D = w.get_distance_matrix(X)

    kmedoids = KMedoids(n_clusters=10, metric="precomputed").fit(D)
    labels = kmedoids.labels_

    print("Rand Index of K-medoids WEHD-optimized classifier: %s" % rand_score(y, labels))

    # re-map the cluster labels so that they match class labels
    labels_remapped = np.zeros_like(labels)
    for i in range(10):
        mask = (labels == i)
        labels_remapped[mask] = mode(y[mask])[0]
    print("accuracy score: %s" % accuracy_score(y, labels_remapped))
    print("confusion matrix: \n%s" % confusion_matrix(y, labels_remapped))

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, learning_rate=10, metric="precomputed")
    result = tsne.fit_transform(D)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=10, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Accent)
    colors = [mapper.to_rgba(label) for label in labels]

    fig = plt.figure()
    plt.scatter(result[:, 0], result[:, 1], c=colors, marker="o", edgecolors="black", picker=True)

    def onpick(event):
        ind = event.ind
        print()
        for i in ind:
            print(y[i])

    fig.canvas.mpl_connect("pick_event", onpick)

    plt.show()
