'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 3: Mining the 20 Newsgroups Dataset with Clustering and Topic Modeling Algorithms
Author: Yuxi (Hayden) Liu
'''


from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

import numpy as np
from matplotlib import pyplot as plt

k = 3
from sklearn.cluster import KMeans
kmeans_sk = KMeans(n_clusters=3, random_state=42)
kmeans_sk.fit(X)
clusters_sk = kmeans_sk.labels_
centroids_sk = kmeans_sk.cluster_centers_

for i in range(k):
    cluster_i = np.where(clusters_sk == i)
    plt.scatter(X[cluster_i, 0], X[cluster_i, 1])
plt.scatter(centroids_sk[:, 0], centroids_sk[:, 1], marker='*', s=200, c='#050505')
plt.show()
