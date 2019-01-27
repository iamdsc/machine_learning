# K-Means Clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import time

# Creating sample data
X, y=make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

# plot the sample points
plt.scatter(X[:,0], X[:,1], c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.show()

# Training the model
# We set number of desired clusters =3
# We set n_init=10 to run k-means clustering algo 10 times independently with diff random centroids
# to choose the final model as the one with lowest SSE
# via max_iter we specify maximum no. of iterations for each single run
# tol controls the tolerance with regard to the changes in the within SSE to declare convergence
km=KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)

tic=time.time()
y_km=km.fit_predict(X)
toc=time.time()
print('Time for training and prediction: ',(toc-tic)*1000,'ms')

# Within cluster SSE
print('Distortion: %.2f' % km.inertia_)

# plotting the clusters with the centroids
plt.scatter(X[y_km==0,0], X[y_km==0,1], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1')
plt.scatter(X[y_km==1,0], X[y_km==1, 1], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2')
plt.scatter(X[y_km==2,0], X[y_km==2, 1], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, marker='*', c='red', edgecolor='black', label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
