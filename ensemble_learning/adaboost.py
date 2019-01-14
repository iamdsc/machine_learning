# AdaBoost Ensemble Classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

# Loading the data
df_wine=pd.read_csv('wine.data', header=None)
df_wine.columns=['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
# drop 1 class
df_wine=df_wine[df_wine['Class label']!=1]
y=df_wine['Class label'].values
X=df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

tree=DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=1)
ada=AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)
tree=tree.fit(X_train, y_train)
y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)
tree_train=accuracy_score(y_train, y_train_pred)
tree_test=accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

ada=ada.fit(X_train, y_train)
y_train_pred=ada.predict(X_train)
y_test_pred=ada.predict(X_test)
ada_train=accuracy_score(y_train, y_train_pred)
ada_test=accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))

# plotting the decision regions
x_min=X_train[:,0].min()-1
x_max=X_train[:,0].max()+1
y_min=X_train[:,1].min()-1
y_max=X_train[:,1].max()+1
xx, yy=np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8,3))
for idx, clf, tt in zip([0,1], [tree,ada], ['Decision Tree', 'AdaBoost']):
    clf.fit(X_train, y_train)
    Z=clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z=Z.reshape(xx.shape)
    axarr[idx].contourf(xx,yy,Z,alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0,0], X_train[y_train==0,1], c='blue',marker='^')
    axarr[idx].scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='red', marker='o')
    axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -0.5, s='OD280/OD315 of diluted wines', ha='center', va='center', fontsize=12)
plt.show()
