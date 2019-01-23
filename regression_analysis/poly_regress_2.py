# Modeling Nonlinear relationships in the housing dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# Loading the data
# Housing dataset
df=pd.read_csv('housing.data.txt', header=None, sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
X=df[['LSTAT']].values # percent lower status of population
y=df['MEDV'].values

regr=LinearRegression()

# create quadratic features
quadratic=PolynomialFeatures(degree=2)
cubic=PolynomialFeatures(degree=3)
X_quad=quadratic.fit_transform(X)
X_cubic=cubic.fit_transform(X)

# fit_features
X_fit=np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr=regr.fit(X, y)
y_lin_fit=regr.predict(X_fit)
linear_r2=r2_score(y, regr.predict(X))

regr=regr.fit(X_quad, y)
y_quad_fit=regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2=r2_score(y, regr.predict(X_quad))

regr=regr.fit(X_cubic, y)
y_cubic_fit=regr.predict(cubic.fit_transform(X_fit))
cubic_r2=r2_score(y, regr.predict(X_cubic))

# plotting results
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2, color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')
plt.show()

# Using log transformation of LSTAT and square root of MEDV for linear fit
# transform features
X_log=np.log(X)
y_sqrt=np.sqrt(y)

# fit features
X_fit=np.arange(X_log.min()-1, X_log.max()+1, 1)[:,np.newaxis]
regr=regr.fit(X_log, y_sqrt)
y_lin_fit=regr.predict(X_fit)
linear_r2=r2_score(y_sqrt, regr.predict(X_log))

# plot results
plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')
plt.show()
