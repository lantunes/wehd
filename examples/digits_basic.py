import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.cluster import rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    digits = datasets.load_digits()

    X = digits.data
    y = digits.target

    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    kmedoids = KMedoids(n_clusters=10, metric="euclidean").fit(X)
    labels = kmedoids.labels_

    print("Rand Index of K-medoids classifier: %s" % rand_score(y, labels))

    # re-map the cluster labels so that they match class labels
    labels_remapped = np.zeros_like(labels)
    for i in range(10):
        mask = (labels == i)
        labels_remapped[mask] = mode(y[mask])[0]
    print("accuracy score: %s" % accuracy_score(y, labels_remapped))
    print("confusion matrix: \n%s" % confusion_matrix(y, labels_remapped))

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, learning_rate=10, metric="euclidean")
    result = tsne.fit_transform(X)

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
