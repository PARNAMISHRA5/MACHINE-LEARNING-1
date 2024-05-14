import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("housing.csv")
df.head(3)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6,n_init=10,max_iter=300,random_state=42,init="random")

df['Cluster'] = kmeans.fit_predict(df)
df["Cluster"] = df["Cluster"].astype("category")
df.head()

sns.relplot(
    x="longitude", y="latitude", hue="Cluster", data=df, height=6,
);

kmeans.inertia_
kmeans.cluster_centers_

sum_se = []
list_k = list(range(1,10))
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sum_se.append(km.inertia_)
plt.figure(figsize=(6,6))
plt.plot(list_k,sum_se,'-o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared error')
plt.title(label="Elbow Method")
plt.show()

from sklearn.metrics import silhouette_score
kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 100,"random_state": 42}
from sklearn.metrics import silhouette_score
silhoutte_coef = []
for k in range(2,11):
    km = KMeans(n_clusters=k,**kmeans_kwargs)
    km.fit(df)
    score = silhouette_score(df,kmeans.labels_)
    silhoutte_coef.append(score)

plt.figure(figsize=(6,6))
plt.plot(range(2,11),silhoutte_coef,'-o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhoutte Coeff')
plt.title(label="Silhoutte Method")
plt.show()
print(silhoutte_coef)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df.values

preprocessor = Pipeline([
    ("scaler",StandardScaler()),
    ("pca",PCA(n_components=2,random_state=42))
])
kmeans = KMeans(n_clusters=6, n_init=10, max_iter=300, random_state=42)


# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', kmeans)
])

# Fit the pipeline to your data
pipeline.fit(df)