
# Copied From Chat GPT 

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Use publication-grade font
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "figure.dpi": 220,
})

def dunn_index(data, labels, centroids):
    unique = np.unique(labels)
    intra, inter = [], []

    for c in unique:
        pts = data[labels == c]
        if len(pts) > 1:
            intra.append(
                np.max(np.linalg.norm(pts[:, None] - pts[None, :], axis=2))
            )

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter.append(np.linalg.norm(centroids[i] - centroids[j]))

    return min(inter) / max(intra)

def xb_index(data, labels, centroids):
    k = len(centroids)

    compact = np.sum([
        np.sum((data[labels == i] - centroids[i]) ** 2)
        for i in range(k)
    ]) / len(data)

    separation = np.min([
        np.linalg.norm(centroids[i] - centroids[j]) ** 2
        for i in range(k)
        for j in range(i + 1, k)
    ])

    return compact / separation


def kmeans_only(csv_path, k=3):
    df = pd.read_csv(csv_path)
    data = df.select_dtypes(include=[np.number]).values

    km = KMeans(n_clusters=k, n_init=12, random_state=42)
    labels = km.fit_predict(data)
    centroids = km.cluster_centers_

    sil  = silhouette_score(data, labels)
    db   = davies_bouldin_score(data, labels)
    ch   = calinski_harabasz_score(data, labels)
    dunn = dunn_index(data, labels, centroids)
    xb   = xb_index(data, labels, centroids)

    print("\n" + "="*70)
    print(f"{' K-MEANS CLUSTERING RESULTS ':^70}")
    print("="*70)
    print(f"{'Silhouette Score':25} : {sil:.4f}")
    print(f"{'Davies-Bouldin Index':25} : {db:.4f}")
    print(f"{'Dunn Index':25} : {dunn:.4f}")
    print(f"{'Xie-Beni Index':25} : {xb:.4f}")
    print(f"{'Calinski-Harabasz':25} : {ch:.4f}")
    print("="*70 + "\n")

    # -------------------------------------------------------------
    # PUBLICATION-READY 3D PCA CLUSTER VISUALIZATION
    # -------------------------------------------------------------

    pca_3d = PCA(n_components=3)
    reduced = pca_3d.fit_transform(data)
    cent = pca_3d.transform(centroids)

    fig = plt.figure(figsize=(10, 7), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    # BLACK BACKGROUND
    fig.patch.set_facecolor("white")   # Figure bg
    ax.set_facecolor("white")          # Axes bg

    # Make texts visible (white)
    ax.title.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.zaxis.label.set_color("black")
    ax.tick_params(colors="black")

    # Soft color palette
    palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd"   # purple
    ]

    # Plot clusters
    for cid in np.unique(labels):
        pts = reduced[labels == cid]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=32,
            alpha=0.85,
            color=palette[cid % len(palette)],
            edgecolors="black",
            linewidth=0.35
        )

    # Plot centroids
    ax.scatter(
        cent[:, 0], cent[:, 1], cent[:, 2],
        s=150,
        color="black",
        marker="o",
        linewidth=1
    )

    # Camera
    ax.view_init(elev=20, azim=30)
    ax.dist = 10

    # Labels
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    # TITLE → moved closer (pad=2)
    ax.set_title(
        "3D PCA Projection of K-Means Clusters",
        fontweight="bold",
        pad=2
    )

    plt.tight_layout()
    plt.show()

    return labels, centroids


if __name__ == "__main__":
    csv_file = "/Users/souvik/Documents/FGOIKmeans/Data-sets/iris.csv"
    kmeans_only(csv_file, k=3)
