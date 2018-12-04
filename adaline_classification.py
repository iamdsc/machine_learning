# Implementing Adaptive Linear Neuron (Adaline) Classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AdalineGD(object):
    """ Adaptive Linear Neuron Classifier """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data. """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_=[]

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y-output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """ Calculate the net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ Compute linear activation """
        return X

    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# Loading the data
df = pd.read_csv('iris.data', header=None)

# select setosa and versicolor
y=df.iloc[0:100, 4].values
y=np.where(y=='Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plotting cost against number of epochs for eta=0.01 and eta=0.0001
##fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
##ada1 = AdalineGD(n_iter=100, eta=0.01).fit(X, y)
##ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
##ax[0].set_xlabel('Epochs')
##ax[0].set_ylabel('log(SSE)')
##ax[0].set_title('Adaline - Learning Rate 0.01')
##
##ada2 = AdalineGD(n_iter=100, eta=0.0001).fit(X, y)
##ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
##ax[1].set_xlabel('Epochs')
##ax[1].set_ylabel('SSE')
##ax[1].set_title('Adaline - Learning Rate 0.0001')
##plt.show()

# feature scaling and standardization
X_std = np.copy(X)
X_std[:,0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()

ada=AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """ Visualize decision boundaries for 2-D datasets"""
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max=X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max=X[:,1].min()-1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0],y=X[y==cl,1],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')
 
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardised]')
plt.ylabel('petal length [standardised]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('SSE')
plt.show()
