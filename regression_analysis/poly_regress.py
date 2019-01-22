# Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dummy data creation
X=np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:,np.newaxis]
y=np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])

lr=LinearRegression()
pr=LinearRegression()

quadratic=PolynomialFeatures(degree=2)
X_quad=quadratic.fit_transform(X)

# fitting simple linear regression model
lr.fit(X, y)
X_fit=np.arange(250, 600, 10)[:,np.newaxis]
y_lin_fit=lr.predict(X_fit)

# fitting multiple regression model on the transformed features for polynomial regression
pr.fit(X_quad, y)
y_quad_fit=pr.predict(quadratic.fit_transform(X_fit))

# plotting the results
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')
plt.show()

# Computing MSE and R^2 score
y_lin_pred=lr.predict(X)
y_quad_pred=pr.predict(X_quad)
print('Training MSE linear: %.3f, quadratic: %.3f' % (mean_squared_error(y, y_lin_pred), mean_squared_error(y, y_quad_pred)))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (r2_score(y, y_lin_pred), r2_score(y, y_quad_pred)))
