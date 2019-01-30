# Hierarchical Agglomerative complete linkage clustering using sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Generating random sample data to work with
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X=np.random.random_sample([5, 3])*10

# Fitting the model
ac=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
labels=ac.fit_predict(X)
print('Cluster labels: %s' % labels)
