from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


def rbf_kernel_pca(X,gamma,n_components):
    # Calculate pairwise squared Euclidean distances in the MxN dimensional dataset
    sq_dists=pdist(X,'sqeuclidean')

    # Convert pairwise distances into a square matrix
    mat_sq_dists=squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K=exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N=K.shape[0]
    one_n=np.ones((N,N))/N
    K=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs
    eigvals,eigvecs=eigh(K)
    eigvals,eigvecs=eigvals[::-1],eigvecs[:,::-1]

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:,i] for i in range(n_components)))

    # Collect the corresponding eigenvalues
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas,lambdas

X,y=make_moons(n_samples=100,random_state=123)
alphas,lambdas=rbf_kernel_pca(X,gamma=15,n_components=1)

# projecting new samples
x_new=X[25]
x_proj=alphas[25]   #original projection
print('Original Projection',x_proj)

def project_x(x_new,X,gamma,alphas,lambdas):
    pair_dist=np.array([np.sum((x_new-row)**2) for row in X])
    k=np.exp(-gamma*pair_dist)
    return k.dot(alphas/lambdas)

x_reproj=project_x(x_new,X,gamma=15,alphas=alphas,lambdas=lambdas)
print('Reprojection',x_reproj)

# visualize the projection on 1st principal component
plt.scatter(alphas[y==0,0],np.zeros((50)),color='red',marker='^',alpha=0.5)
plt.scatter(alphas[y==1,0],np.zeros((50)),color='blue',marker='o',alpha=0.5)

plt.scatter(x_proj,0,color='black',label='original projection of point X[25]',marker='^',s=100)
plt.scatter(x_reproj,0,color='green',label='remapped point X[25]',marker='x',s=500)
plt.legend(scatterpoints=1)
plt.show()
