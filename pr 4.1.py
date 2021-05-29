import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs #набор данных

#набор данных
X, y_true = make_blobs(n_samples = 500, centers = 4, cluster_std = 0.40, random_state = 0) 

#визуализация набора данных
plt.scatter(X[:, 0], X[:, 1], s = 50)

kmeans = KMeans(n_clusters = 4)#описываем модель kmeans в качестве алгоритма KMeans

#обучение модели K-средних
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 50, cmap = 'viridis') #визуализация

#центры
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.10)

plt.show()