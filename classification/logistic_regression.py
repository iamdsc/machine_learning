# Implementing Logistic Regression classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

class LogisticRegressionGD(object):
    """ Logisitic Regression Classifier using Gradient Descent """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # logistic cost
            cost = (-y.dot(np.log(output))-((1-y).dot(np.log(1-output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """ Calculate the net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """ Compute logistic sigmoid activation """
        return 1/(1+np.exp(-np.clip(z,-250,250)))

    def predict(self, X):
        """ Return the class label after unit step"""
        return np.where(self.net_input(X)>=0.0, 1, 0)

# Loading the data
iris = datasets.load_iris()
X = iris.data[:,[0,2]]
y=iris.target

X_01_subset = X[(y==0) | (y==1)]
y_01_subset = y[(y==0) | (y==1)]

lrgd = LogisticRegressionGD(eta = 0.05, n_iter=1000, random_state=1)
lrgd.fit(X_01_subset, y_01_subset)

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
 
plot_decision_regions(X=X_01_subset, y=y_01_subset, classifier=lrgd)
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
