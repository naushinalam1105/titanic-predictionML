import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix

# 1. Load kaggle csv
df = pd.read_csv("Iris.csv")

# drop 'Id' column if present
if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)

# 2. Separate features (X) and ground truth target (y)
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_cols]

# convert string species names to integer labels for comparison
le = LabelEncoder()
y_true = le.fit_transform(df['Species'])
target_names = le.classes_

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA to reduce 4D features to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Preserved: {sum(pca.explained_variance_ratio_):.2%}")

# 5. Apply K-Means clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# 6. Evaluation metrics
ari = adjusted_rand_score(y_true, y_kmeans)
print(f"\nAdjusted Rand Index (ARI Score): {ari:.4f}")
print("\nContingency Matrix (True Species vs Cluster IDs):")
print(confusion_matrix(y_true, y_kmeans))

# 7. Visualization: True Labels vs K-Means Clusters
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Ground Truth
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', edgecolor='k', s=50)
axes[0].set_title("Ground Truth Classes (PCA Reduced)")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
handles, _ = scatter1.legend_elements()
axes[0].legend(handles, target_names, title="Species")

# Plot 2: K-Means Predicted Clusters
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='plasma', edgecolor='k', s=50)
centroids_pca = pca.transform(kmeans.cluster_centers_)
axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
axes[1].set_title("K-Means Clusters (k=3)")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].legend()

plt.tight_layout()
plt.savefig("iris_clusters_vs_true.png")
plt.show()
print("Plot saved to iris_clusters_vs_true.png")