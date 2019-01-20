# Regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler


# Housing dataset
# No. of samples: 506
# No. of explanatory variables: 13

df=pd.read_csv('housing.data.txt', header=None, sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.head())

# Creating a scatter plot matrix to visualize pair-wise correlations
# between different features in this dataset in one place
cols=['LSTAT','INDUS','NOX','RM','MEDV']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

# plotting correlation matrix array as heat map
cm=np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.show()

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self, X, y):
        self.w_=np.zeros(1+X.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            output=self.net_input(X)
            errors=(y-output)
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self, X):
        return self.net_input(X)
    
X=df[['RM']].values
y=df['MEDV'].values

# Standardizing the variables for better convergence of the GD algorithm
sc_x=StandardScaler()
sc_y=StandardScaler()
X_std=sc_x.fit_transform(X)
y_std=sc_y.fit_transform(y[:,np.newaxis]).flatten()

# Training the model
lr=LinearRegressionGD()
lr.fit(X_std,y_std)

# plotting cost as a function of no. of epochs over training dataset
sns.reset_orig() # resets matplotlib style
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X,y,model):
    """ Helper function to plot scatterplot of training samples and add regression line """
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

lin_regplot(X_std,y_std,lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

# Pridicting price in original scale
num_rooms_std=sc_x.transform(np.array([5.0]).reshape(1,-1))
price_std=lr.predict(num_rooms_std)
print('Price in $1000s for 5 rooms: %.3f' % sc_y.inverse_transform(price_std))
