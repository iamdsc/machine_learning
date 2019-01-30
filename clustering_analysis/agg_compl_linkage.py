# Hierarchical Agglomerative complete linkage clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# Generating random sample data to work with
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X=np.random.random_sample([5, 3])*10
df=pd.DataFrame(X, columns=variables, index=labels)
#print(df)

# Computing Distance matrix
row_dist=pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
#print(row_dist)

# Applying complete linkage agglomeration to get linkage matrix
# We can use condensed distance matrix or initial data array
row_clusters=linkage(df.values, method='complete', metric='euclidean')

# Turning clustering results into a pandas DataFrame
cl_df=pd.DataFrame(row_clusters, columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'], index=['cluster %d' % (i+1) for i in range(row_clusters.shape[0])])
# each row represents one merge
# 1st and 2nd col denote most dissimilar members in each cluster and 3rd denotes dist between them
# last column denotes count of members in each cluster
# print(cl_df)

# Visualizing the results in the form of a Dendrogram
row_dendr=dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()

# Attaching dendrograms to a heat map
fig=plt.figure(figsize=(8,8), facecolor='white')
axd=fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr=dendrogram(row_clusters, orientation='left')

# Reorder data in df according to clustering labels
df_rowclust=df.iloc[row_dendr['leaves'][::-1]]

# Constructing heat map from reordered df
axm=fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax=axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels(['']+list(df_rowclust.columns))
axm.set_yticklabels(['']+list(df_rowclust.index))
plt.show()
