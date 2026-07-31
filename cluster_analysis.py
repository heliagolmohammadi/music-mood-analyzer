import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data with extracted audio features
df = pd.read_csv("tracks_with_features.csv")

# Select the numeric features we extracted earlier
feature_columns = ["tempo", "energy", "brightness", "noisiness", "tonality"]
X = df[feature_columns].copy()

# Standardize features (important: puts all features on the same scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try clustering into groups (let's start with 4 groups/clusters)
NUM_CLUSTERS = 4
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Save the result
df.to_csv("tracks_with_clusters.csv", index=False, encoding="utf-8-sig")

print(f"Done! Tracks grouped into {NUM_CLUSTERS} clusters.")
print("\nNumber of tracks per cluster:")
print(df["cluster"].value_counts().sort_index())

print("\nAverage feature values per cluster:")
print(df.groupby("cluster")[feature_columns].mean())

# Show a few example tracks from each cluster
print("\nSample tracks from each cluster:")
for c in sorted(df["cluster"].unique()):
    print(f"\n--- Cluster {c} ---")
    sample = df[df["cluster"] == c][["name", "artist_name", "tempo", "energy"]].head(3)
    print(sample.to_string(index=False))

# Simple visualization: plot tempo vs energy, colored by cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["tempo"], df["energy"], c=df["cluster"], cmap="viridis")
plt.xlabel("Tempo (BPM)")
plt.ylabel("Energy")
plt.title("Music Clusters: Tempo vs Energy")
plt.colorbar(scatter, label="Cluster")
plt.savefig("cluster_plot.png")
print("\nSaved a visualization to cluster_plot.png")