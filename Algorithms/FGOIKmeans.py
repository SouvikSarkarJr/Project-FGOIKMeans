#   ------------------- सांब सदाशिव -------------------  #

#   कृष्णाय वासुदेवाय हरये परमात्मने प्रणतः क्लेशनाशाय गोविन्दाय नमो नमः   #

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#  Fungal Growth Optimization (FGO)

def compute_fitness(centroids, data):
    labels = assign_clusters(data, centroids)

    compactness = 0.0
    for c in range(len(centroids)):
        pts = data[labels == c]
        if pts.size > 0:
            compactness += np.sum((pts - centroids[c])**2)

    separation = np.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d < separation:
                separation = d

    if separation == 0:
        separation = 1e-10

    if len(np.unique(labels)) > 1:
        sil = silhouette_score(data, labels)
    else:
        sil = 0.0

    return (compactness / (separation**2)) - sil


def generate_random_centroids(k, data):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    return mins + np.random.rand(k, data.shape[1]) * (maxs - mins)


def fungal_growth_optimization(
    data,
    init_centroids,
    k=3,
    hyphae_count=20,
    iterations=50,
    growth_rate=0.05,
    branching_rate=0.3
):

    hyphae = [init_centroids.copy() for _ in range(hyphae_count)]

    best_hypha = hyphae[0]
    best_fitness = compute_fitness(best_hypha, data)

    for it in range(iterations):
        new_population = []

        for hypha in hyphae:
            fitness = compute_fitness(hypha, data)

            if fitness < best_fitness:
                best_fitness = fitness
                best_hypha = hypha.copy()

            direction = best_hypha - hypha
            grown_hypha = hypha + growth_rate * direction * np.random.rand(*hypha.shape)
            new_population.append(grown_hypha)

            if np.random.rand() < branching_rate:
                branch = hypha + (np.random.randn(*hypha.shape) * 0.15)
                new_population.append(branch)

        fitnesses = [compute_fitness(h, data) for h in new_population]
        sorted_idx = np.argsort(fitnesses)

        hyphae = [new_population[i] for i in sorted_idx[:hyphae_count]]

        print(f"FGO Iteration {it+1}/{iterations} | Best Fitness = {best_fitness:.4f}")

    return best_hypha


def kmeans_with_init(data, init_centroids, max_iter=100):
    k = init_centroids.shape[0]
    centroids = init_centroids.copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            data[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c]
            for c in range(k)
        ])

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return labels, centroids


# CLUSTERING INDEXES

def dunn_index(data, labels, centroids):
    unique_clusters = np.unique(labels)
    intra_dists = []
    inter_dists = []

    for c in unique_clusters:
        cluster_points = data[labels == c]
        if len(cluster_points) > 1:
            intra = np.max(np.linalg.norm(
                cluster_points[:, None] - cluster_points[None, :], axis=2
            ))
            intra_dists.append(intra)

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

    return min(inter_dists) / max(intra_dists)


def xb_index(data, labels, centroids):
    k = len(centroids)

    compactness = np.sum([
        np.sum((data[labels == i] - centroids[i])**2)
        for i in range(k)
    ]) / len(data)

    sep = np.min([
        np.linalg.norm(centroids[i] - centroids[j])**2
        for i in range(k)
        for j in range(i + 1, k)
    ])

    return compactness / sep


def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
    return np.argmin(distances, axis=1)

# 3D VISUALIZATION

def plot_3d_pca(data, labels, centroids):
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)
    centroids_pca = pca.transform(centroids)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        data_pca[:, 0],
        data_pca[:, 1],
        data_pca[:, 2],
        c=labels,
        cmap='viridis',
        s=40,
        alpha=0.7
    )

    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        centroids_pca[:, 2],
        c='red',
        s=250,
        marker='X',
        label='Centroids'
    )

    ax.set_title("3D PCA Projection (FGO + KMeans)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    plt.show()


def hybrid_fgo_kmeans(csv_path, k=3, fgo_iters=50):
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=[np.number])
    data = numeric_df.values

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km_labels = km.fit_predict(data)
    km_centroids = km.cluster_centers_

    fgo_centroids = fungal_growth_optimization(
        data, km_centroids, k=k, iterations=fgo_iters
    )

    labels = assign_clusters(data, fgo_centroids)
    final_centroids = fgo_centroids

    silhouette = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)
    ch = calinski_harabasz_score(data, labels)
    dunn = dunn_index(data, labels, final_centroids)
    xb = xb_index(data, labels, final_centroids)

    print("\n================== CLUSTERING RESULT ==================")
    print(f"Silhouette Score      : {silhouette:.4f}")
    print(f"Davies-Bouldin Index  : {db:.4f}")
    print(f"Dunn Index            : {dunn:.4f}")
    print(f"Xie-Beni Index        : {xb:.4f}")
    print(f"Calinski-Harabasz     : {ch:.4f}")
    print("========================================================\n")
 
    plot_3d_pca(data, labels, final_centroids)

    return labels, final_centroids


if __name__ == "__main__":

    csv_file = "/Users/souvik/Documents/FGOIKmeans/Data-sets/iris.csv"  # CSV File Path
    hybrid_fgo_kmeans(csv_file, k=3, fgo_iters=50)
