# Iris Flower Clustering Project
# Week 3 - Unsupervised Learning, Clustering & Dimensionality Reduction

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# Load dataset
df = pd.read_csv("Iris.csv")

# Select the four measurement features.
# Id is only an identifier and Species is the true label.
feature_cols = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

X = df[feature_cols]
y = df["Species"]

# -------------------------------------------------
# 1. Apply K-Means clustering with k = 3
# -------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print("K-MEANS CLUSTER CENTERS")
print(pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols))

print("\nCluster sizes:")
print(pd.Series(clusters).value_counts().sort_index())

# -------------------------------------------------
# 2. Visualize K-Means clusters
# -------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    X["PetalLengthCm"],
    X["PetalWidthCm"],
    c=clusters,
    s=50,
    alpha=0.75
)

plt.scatter(
    kmeans.cluster_centers_[:, 2],
    kmeans.cluster_centers_[:, 3],
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset - K-Means Clusters (k=3)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 3. Apply PCA to reduce the dataset to 2D
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    s=50,
    alpha=0.75
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Dataset - PCA (2D)")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 4. Find explained variance ratio
# -------------------------------------------------
print("\nPCA EXPLAINED VARIANCE RATIO")
print(f"PC1: {pca.explained_variance_ratio_[0]:.4%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.4%}")
print(
    f"Total (PC1 + PC2): "
    f"{pca.explained_variance_ratio_.sum():.4%}"
)

# -------------------------------------------------
# 5. Compare predicted clusters with true labels
# -------------------------------------------------
comparison = pd.crosstab(
    pd.Series(clusters, name="Predicted Cluster"),
    pd.Series(y, name="True Species")
)

print("\nPREDICTED CLUSTERS VS TRUE LABELS")
print(comparison)

ari = adjusted_rand_score(y, clusters)
print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")

# Save cluster assignments
output = df.copy()
output["Cluster"] = clusters
output.to_csv("Iris_KMeans_Results.csv", index=False)

print("\nResults saved to Iris_KMeans_Results.csv")


# -------------------------------------------------
# 6. Project conclusion
# -------------------------------------------------
print("\nPROJECT CONCLUSION")
print("K-Means clustering with k=3 grouped the Iris dataset into three clusters.")
print(f"The Adjusted Rand Index (ARI) was {ari:.4f}, showing meaningful agreement with the true species labels.")
print(f"The first two principal components explained {pca.explained_variance_ratio_.sum():.2%} of the total variance.")
print("Overall, the project demonstrates unsupervised learning and dimensionality reduction on the Iris dataset.")
print("Note: K-Means cluster labels are arbitrary numerical identifiers and do not directly represent Iris species.")
