# Implemeting Linear Regression using sklearn
# Regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Housing dataset
# No. of samples: 506
# No. of explanatory variables: 13

df=pd.read_csv('housing.data.txt', header=None, sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
#print(df.head())

X=df[['RM']].values
y=df['MEDV'].values

slr=LinearRegression()
slr.fit(X,y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

# plotting the regression line
def lin_regplot(X,y,model):
    """ Helper function to plot scatterplot of training samples and add regression line """
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

lin_regplot(X,y,slr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in  $1000s [MEDV] (standardized)')
plt.show()
