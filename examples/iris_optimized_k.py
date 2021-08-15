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


"""
Here we take the Iris dataset, consisting of 150 labelled examples with 4 continuous features each. There are 3 classes.
We use K-medoids clustering with k=3 and a weighted Euclidean-Hamming distance. We optimize the weights of the metric,  
and the number of clusters, using Evolution Strategies. Thus, this is an example of unsupervised metric learning.

The plot contains the examples reduced to 2 dimensions using t-SNE (with the same metric). The color of each point 
represents the label assigned to the point by clustering, while the shape (circle, triangle, square) represents the 
true class. There is less confusion between the square and triangle classes when compared to using the standard, 
unweighted Euclidean metric. That is, K-medoids clustering with the optimized weighted Euclidean-Hamming metric achieves 
an accuracy score of ~0.96 in a classification setting, as opposed to a score of 0.90 obtained using a K-medoids 
classifier with a standard, unweighted Euclidean metric.

Here, we choose to minimize the Blur Ratio: the sum of the cluster variances divided by the total variance. This is the 
objective used in the paper:

Gupta, A. A., Foster, D. P., & Ungar, L. H. (2008). Unsupervised distance metric learning using predictability. 
Technical Reports (CIS), 885.

Here, We don't assume the number of clusters. We add a term to our objective that penalizes a higher number of clusters.
We don't assume we know the number of clusters, but instead provide a range of guesses, numbers in [2, 6], and the
optimization discovers that the ideal number of clusters is 3.
"""
if __name__ == '__main__':
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)


    def model(para):
        weights = [para["w1"], para["w2"], para["w3"], para["w4"]]
        w = WEHD(categorical_indices=[], weights=weights)

        D = w.get_distance_matrix(X)

        kmedoids = KMedoids(n_clusters=int(para["k"]), metric="precomputed").fit(D)
        labels = kmedoids.labels_

        clusters = set(labels)

        SSC = 0  # the within cluster variance
        for label in clusters:
            X_indices = [i for i, v in enumerate(labels) if v == label]
            SSC += cluster_variance(X_indices, D)

        SST = cluster_variance(list(range(len(X))), D)  # the total variance

        # adding a term that attempts to get the number of clusters being ideal, so that we don't get too many clusters
        # TODO how do we determine the factor 0.05?
        return -(SSC / SST + 0.05*len(clusters))  # we want to minimize SSC/SST


    search_space = {
        "k": [1, 2, 3, 4, 5, 6],
        "w1": np.linspace(0, 1, 21),
        "w2": np.linspace(0, 1, 21),
        "w3": np.linspace(0, 1, 21),
        "w4": np.linspace(0, 1, 21),
    }

    opt = EvolutionStrategyOptimizer(search_space)
    opt.search(model, n_iter=2500)

    best_params = opt.best_para
    weights = [best_params["w1"], best_params["w2"], best_params["w3"], best_params["w4"]]
    w = WEHD(categorical_indices=[], weights=weights)
    D = w.get_distance_matrix(X)

    kmedoids = KMedoids(n_clusters=int(best_params["k"]), metric="precomputed").fit(D)
    labels = kmedoids.labels_

    clusters = set(labels)
    print("num clusters: %s" % len(clusters))

    print("Rand Index of K-medoids WEHD-optimized classifier: %s" % rand_score(y, labels))

    # re-map the cluster labels so that they match class labels
    labels_remapped = np.zeros_like(labels)
    for i in range(3):
        mask = (labels == i)
        labels_remapped[mask] = mode(y[mask])[0]
    print("accuracy score: %s" % accuracy_score(y, labels_remapped))
    print("confusion matrix: \n%s" % confusion_matrix(y, labels_remapped))

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, learning_rate=10, metric="precomputed")
    result = tsne.fit_transform(D)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(clusters), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Accent)
    colors = [mapper.to_rgba(label) for label in labels]

    # the first 50 are true label 0, the next 50 are true label 1, and the last 50 are true label 2
    plt.scatter(result[:50, 0], result[:50, 1], c=colors[:50], marker="o", edgecolors='black')
    plt.scatter(result[50:100, 0], result[50:100, 1], c=colors[50:100], marker="^", edgecolors='black')
    plt.scatter(result[100:150, 0], result[100:150, 1], c=colors[100:150], marker="s", edgecolors='black')

    plt.show()
