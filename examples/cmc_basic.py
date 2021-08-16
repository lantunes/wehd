import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from wehd import WEHD
from scipy.stats import mode
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.cluster import rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
import warnings

warnings.filterwarnings("ignore")


"""
The UCI Contraceptive Dataset (i.e. "cmc") is difficult to classify, according to Lim, T. S., Loh, W. Y., & Shih, Y. S. 
(2000). A comparison of prediction accuracy, complexity, and training time of thirty-three old and new classification 
algorithms. Machine learning, 40(3), 203-228:

"The most difficult to classify are cmc, cmc+, and tae+, with minimum error rates greater than 0.4."
"""
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

    weights = np.ones(9)  # equal weight to all features
    w = WEHD(categorical_indices=categorical_features, weights=weights / np.sum(weights))
    D = w.get_distance_matrix(X)

    kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed").fit(D)
    labels = kmedoids.labels_

    print("Rand Index of K-medoids classifier: %s" % rand_score(y, labels))

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
