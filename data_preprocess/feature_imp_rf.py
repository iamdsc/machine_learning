import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import matplotlib.pyplot as plt


df_wine=pd.read_csv('wine.data',header=None)
df_wine.columns=['Class label','Alcohol','Malic Acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

## Train - Test split
X, y=df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)

feat_labels=df_wine.columns[1:]
forest=RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train,y_train)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f"%(f+1,30,feat_labels[indices[f]],importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()

sfm=SelectFromModel(forest,threshold=0.1,prefit=True)
X_selected=sfm.transform(X_train)
print('Number of samples that meet this criterion:',X_selected.shape[0])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f"%(f+1,30,feat_labels[indices[f]],importances[indices[f]]))
