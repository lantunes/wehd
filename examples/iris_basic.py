from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.cluster import rand_score
from scipy.stats import mode
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


"""
Here we take the Iris dataset, consisting of 150 labelled examples with 4 continuous features each. There are 3 classes.
We use K-medoids clustering with k=3 and a standard euclidean metric.

The plot contains the examples reduced to 2 dimensions using t-SNE (with the same metric). The color of each point 
represents the label assigned to the point by clustering, while the shape (circle, triangle, square) represents the 
true class. There is some confusion between the square and triangle classes. That is, K-medoids clustering with the 
Euclidean metric is unable to completely discover the true classes. Using K-medoids as a classifier, we achieve a 0.90 
accuracy score.
"""
if __name__ == '__main__':
    n_clusters = 3

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    kmedoids = KMedoids(n_clusters=n_clusters, metric="euclidean").fit(X)
    labels = kmedoids.labels_

    print("Rand Index of K-medoids classifier: %s" % rand_score(y, labels))

    # re-map the cluster labels so that they match class labels
    labels_remapped = np.zeros_like(labels)
    for i in range(3):
        mask = (labels == i)
        labels_remapped[mask] = mode(y[mask])[0]
    print("accuracy score: %s" % accuracy_score(y, labels_remapped))
    print("confusion matrix: \n%s" % confusion_matrix(y, labels_remapped))

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, learning_rate=10, metric="euclidean")
    result = tsne.fit_transform(X)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_clusters, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Accent)
    colors = [mapper.to_rgba(label) for label in labels]

    # the first 50 are true label 0, the next 50 are true label 1, and the last 50 are true label 2
    plt.scatter(result[:50, 0], result[:50, 1], c=colors[:50], marker="o", edgecolors='black')
    plt.scatter(result[50:100, 0], result[50:100, 1], c=colors[50:100], marker="^", edgecolors='black')
    plt.scatter(result[100:150, 0], result[100:150, 1], c=colors[100:150], marker="s", edgecolors='black')

    plt.show()
