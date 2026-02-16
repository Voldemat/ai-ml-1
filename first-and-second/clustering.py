import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

data = np.array([
    [50, 25000],
    [60, 30000],
    [120, 60000],
    [130, 65000],
    [80, 40000]
])

kmeans = KMeans(n_clusters=2, random_state=0)

clusters = kmeans.fit_predict(data)
plt.scatter(data[:,0], data[:, 1], c=clusters)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House clusters")
plt.show()
