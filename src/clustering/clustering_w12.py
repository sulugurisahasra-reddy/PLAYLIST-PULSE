from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "dataset.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading Spotify dataset...")

    df = pd.read_csv(DATA)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Numeric Spotify features
    features = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    # Keep only columns that exist
    features = [col for col in features if col in df.columns]

    print("\nFeatures used for clustering:")
    print(features)

    X = df[features].apply(pd.to_numeric, errors="coerce")

    # Remove missing values
    X = X.dropna()

    # Limit size for faster clustering
    if len(X) > 20000:
        X = X.sample(20000, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nRows used for clustering: {len(X)}")

    return X_scaled


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, X, labels):

    # DBSCAN marks noise as -1
    valid = labels != -1

    X_valid = X[valid]
    labels_valid = labels[valid]

    number_of_clusters = len(set(labels_valid))

    if number_of_clusters < 2:

        print(f"{name}: Not enough clusters for evaluation.")

        return {
            "Model": name,
            "Clusters": number_of_clusters,
            "Silhouette": np.nan,
            "Davies_Bouldin": np.nan
        }

    sample_size = min(len(X_valid), 2000)

    silhouette = silhouette_score(
        X_valid,
        labels_valid,
        sample_size=sample_size,
        random_state=42
    )

    davies = davies_bouldin_score(
        X_valid,
        labels_valid
    )

    print(f"\n{name}")
    print(f"Clusters: {number_of_clusters}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Score: {davies:.4f}")

    return {
        "Model": name,
        "Clusters": number_of_clusters,
        "Silhouette": round(silhouette, 4),
        "Davies_Bouldin": round(davies, 4)
    }


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 60)
    print("WEEK 12 - SPOTIFY CLUSTERING")
    print("=" * 60)

    X = load_data()

    # ========================================================
    # 1. K-MEANS
    # ========================================================

    print("\nRunning K-Means...")

    k_values = range(2, 9)

    inertias = []
    silhouette_values = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        inertias.append(model.inertia_)

        score = silhouette_score(
            X,
            labels,
            sample_size=min(len(X), 2000),
            random_state=42
        )

        silhouette_values.append(score)

        print(
            f"K={k} | "
            f"Inertia={model.inertia_:.2f} | "
            f"Silhouette={score:.4f}"
        )

    # Best K based on silhouette
    best_k = list(k_values)[int(np.argmax(silhouette_values))]

    print(f"\nBest K selected: {best_k}")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(X)

    # ========================================================
    # 2. K-MEANS ELBOW GRAPH
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        inertias,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("W12 - K-Means Elbow Curve")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUT / "W12_KMeans_Elbow.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 3. SILHOUETTE GRAPH
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        silhouette_values,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("W12 - K-Means Silhouette Analysis")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUT / "W12_KMeans_Silhouette.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 4. AGGLOMERATIVE CLUSTERING
    # ========================================================

    print("\nRunning Agglomerative Clustering...")

    agglomerative = AgglomerativeClustering(
        n_clusters=best_k
    )

    agg_labels = agglomerative.fit_predict(X)

    # ========================================================
    # 5. DBSCAN
    # ========================================================

    print("\nRunning DBSCAN...")

    dbscan = DBSCAN(
        eps=1.2,
        min_samples=10
    )

    dbscan_labels = dbscan.fit_predict(X)

    # ========================================================
    # 6. EVALUATION
    # ========================================================

    results = []

    results.append(
        evaluate_model(
            "K-Means",
            X,
            kmeans_labels
        )
    )

    results.append(
        evaluate_model(
            "Agglomerative",
            X,
            agg_labels
        )
    )

    results.append(
        evaluate_model(
            "DBSCAN",
            X,
            dbscan_labels
        )
    )

    results_df = pd.DataFrame(results)

    print("\n================ RESULTS ================")
    print(results_df)

    results_df.to_csv(
        OUT / "W12_Clustering_Results.csv",
        index=False
    )

    # ========================================================
    # 7. PCA VISUALIZATION
    # ========================================================

    print("\nCreating PCA cluster visualization...")

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans_labels,
        s=10,
        alpha=0.6
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("W12 - K-Means Spotify Clusters")

    plt.tight_layout()

    plt.savefig(
        OUT / "W12_KMeans_Clusters.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 8. AGGLOMERATIVE PCA
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=agg_labels,
        s=10,
        alpha=0.6
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("W12 - Agglomerative Clusters")

    plt.tight_layout()

    plt.savefig(
        OUT / "W12_Agglomerative_Clusters.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 9. DBSCAN PCA
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=dbscan_labels,
        s=10,
        alpha=0.6
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("W12 - DBSCAN Clusters")

    plt.tight_layout()

    plt.savefig(
        OUT / "W12_DBSCAN_Clusters.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 10. DENDROGRAM
    # ========================================================

    print("\nCreating dendrogram...")

    # Use a smaller sample for dendrogram
    sample_size = min(len(X), 500)

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(X),
        size=sample_size,
        replace=False
    )

    X_sample = X[indices]

    linkage_matrix = linkage(
        X_sample,
        method="ward"
    )

    plt.figure(figsize=(12, 6))

    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90
    )

    plt.xlabel("Samples / Clusters")
    plt.ylabel("Distance")
    plt.title("W12 - Hierarchical Clustering Dendrogram")

    plt.tight_layout()

    plt.savefig(
        OUT / "W12_Dendrogram.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("WEEK 12 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")
    print("W12_KMeans_Elbow.png")
    print("W12_KMeans_Silhouette.png")
    print("W12_KMeans_Clusters.png")
    print("W12_Agglomerative_Clusters.png")
    print("W12_DBSCAN_Clusters.png")
    print("W12_Dendrogram.png")
    print("W12_Clustering_Results.csv")

    print(f"\nAll files saved in:\n{OUT}")


if __name__ == "__main__":
    run()