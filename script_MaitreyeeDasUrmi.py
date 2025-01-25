import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv('wholesale_customers.csv')
print(df.head())

X = df.iloc[:, 2:] 
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)


silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

kmeans = KMeans(random_state=42, n_clusters=3)
kmeans.fit(X_scaled)

labels = kmeans.labels_

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
for cluster in np.unique(labels):
    cluster_data = X_pca[labels == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}', alpha=0.6)

centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100,linewidths=3, color='red', label=centroids)
plt.title("Wholesale Customer Clusters")
plt.xlabel("PCA_1")
plt.ylabel("PCA_2")
plt.legend()
plt.grid()
plt.show()

