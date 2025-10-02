import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv("data_415.csv")

names = data["FirstLast"]

# drop columns
clustering_data = data.drop(columns=["Timestamp", "FirstLast", "ModelName", "AI", "Month", "Mobile", "Shower", "MostExcitedFor", "Color","FirstTimeHere","Irvine"])

# clusters based on:
## Year,Major,OS,Language,Genre,Meme,Experience

# encode
label_encoders = {}
for col in clustering_data.columns:
    if clustering_data[col].dtype == 'object':
        le = LabelEncoder()
        clustering_data[col] = le.fit_transform(clustering_data[col])
        label_encoders[col] = le

# kNN, k = 10
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(clustering_data)

# stick clusters to names
output = pd.DataFrame({
    "Name": names,
    "Cluster": clusters
})

print(output)

# reduce features to 2D using PCA for visualization !!
pca = PCA(n_components=2)
reduced = pca.fit_transform(clustering_data)

# clusters
plt.figure(figsize=(12, 8))
plt.scatter(reduced[:,0], reduced[:,1], c=clusters, cmap='tab10', s=80, alpha=0.8)

# names to plot
for i, name in enumerate(names):
    plt.text(reduced[i,0]+0.02, reduced[i,1]+0.02, name, fontsize=3)

plt.title("Board Members Clustered (k=5)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster ID")
plt.show()
