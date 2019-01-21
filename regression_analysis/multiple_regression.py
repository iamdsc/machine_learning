# Multiple Regression Model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the data
# Housing dataset
# No. of samples: 506
# No. of explanatory variables: 13

df=pd.read_csv('housing.data.txt', header=None, sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
X=df.iloc[:,:-1].values
y=df['MEDV'].values

# Performing train test split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)

# Training the model
slr=LinearRegression()
slr.fit(X_train, y_train)

y_train_pred=slr.predict(X_train)
y_test_pred=slr.predict(X_test)

# Plotting Residual plots
plt.scatter(y_train_pred, y_train_pred-y_train, c='steelblue', marker='o', edgecolor='white', label='Training Data')
plt.scatter(y_test_pred, y_test_pred-y_test, c='limegreen', marker='s', edgecolor='white', label='Test Data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

# Computing the Mean Squared Error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

# Computing the Coefficient of Determination (R^2)
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
