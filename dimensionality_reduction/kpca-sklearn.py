from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


X,y=make_moons(n_samples=100,random_state=123)
scikit_kpca=KernelPCA(n_components=2,kernel='rbf',gamma=15)
X_skpca=scikit_kpca.fit_transform(X)

# plotting transformed half-moon shape data onto first 2 principal components
plt.scatter(X_skpca[y==0,0],X_skpca[y==0,1],color='red',marker='^',alpha=0.5)
plt.scatter(X_skpca[y==1,0],X_skpca[y==1,1],color='blue',marker='o',alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
