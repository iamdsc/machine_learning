# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the data
# Housing dataset
df=pd.read_csv('housing.data.txt', header=None, sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X=df[['LSTAT']].values
y=df['MEDV'].values

# fit the model
tree=DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

def lin_regplot(X,y,model):
    """ Helper function to plot scatterplot of training samples and add regression line """
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

# plot the results
sort_idx=X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% loer status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()
