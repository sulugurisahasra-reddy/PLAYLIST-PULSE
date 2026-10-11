from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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

print("=" * 60)
print("SPOTIFY K-MEANS CLUSTERING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA)

print(f"Dataset shape: {df.shape}")

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# SELECT NUMERIC FEATURES
# ============================================================

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

# Keep only features available in the dataset
features = [feature for feature in features if feature in df.columns]

print("\nFeatures used:")
print(features)


X = df[features].copy()

# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Remove missing values
X = X.dropna()

print(f"\nRows after removing missing values: {len(X)}")


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardization completed.")


# ============================================================
# ELBOW METHOD
# ============================================================

print("\nRunning Elbow Method...")

k_values = range(2, 11)

inertias = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)

    print(f"K = {k}, Inertia = {model.inertia_:.2f}")


# Plot Elbow Curve

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUT / "KMeans_Elbow.png",
    dpi=150
)

plt.close()


# ============================================================
# SILHOUETTE ANALYSIS
# ============================================================

print("\nRunning Silhouette Analysis...")

silhouette_scores = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=min(len(X_scaled), 2000),
        random_state=42
    )

    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.4f}"
    )


# Plot Silhouette Scores

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Analysis")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUT / "KMeans_Silhouette.png",
    dpi=150
)

plt.close()


# ============================================================
# SELECT BEST K
# ============================================================

best_k = list(k_values)[
    int(np.argmax(silhouette_scores))
]

best_score = max(silhouette_scores)

print("\nBest number of clusters:", best_k)
print("Best Silhouette Score:", round(best_score, 4))


# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

print("\nTraining final K-Means model...")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)


# ============================================================
# ADD CLUSTER LABELS
# ============================================================

clustered_data = X.copy()

clustered_data["Cluster"] = cluster_labels


# Save clustered data

clustered_data.to_csv(
    OUT / "KMeans_Clustered_Data.csv",
    index=False
)


# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

cluster_counts = (
    clustered_data["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nCluster distribution:")
print(cluster_counts)


# Plot Cluster Distribution

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Number of Tracks")
plt.title("Spotify Track Cluster Distribution")

plt.tight_layout()

plt.savefig(
    OUT / "KMeans_Cluster_Distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# PCA VISUALIZATION
# ============================================================

print("\nCreating PCA visualization...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    s=10,
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    f"Spotify K-Means Clustering (K={best_k})"
)

plt.tight_layout()

plt.savefig(
    OUT / "KMeans_PCA_Clusters.png",
    dpi=150
)

plt.close()


# ============================================================
# CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    clustered_data
    .groupby("Cluster")[features]
    .mean()
)

print("\nCluster summary:")
print(cluster_summary)


cluster_summary.to_csv(
    OUT / "KMeans_Cluster_Summary.csv"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("CLUSTERING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

print("KMeans_Elbow.png")
print("KMeans_Silhouette.png")
print("KMeans_Cluster_Distribution.png")
print("KMeans_PCA_Clusters.png")
print("KMeans_Clustered_Data.csv")
print("KMeans_Cluster_Summary.csv")

print("\nFiles saved in:")
print(OUT)