import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


# Loading the data
df=pd.read_csv('wdbc.data',header=None)
X=df.loc[:,2:].values
y=df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)

# Performing train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)

pipe_svc=make_pipeline(StandardScaler(), SVC(random_state=1))
param_range=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},
                   {'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}]
gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=1)
gs=gs.fit(X_train, y_train)

print('Grid Search Best Score',gs.best_score_)
print('Grid Search Best Parameters',gs.best_params_)

clf=gs.best_estimator_
clf.fit(X_train,y_train)

print('Test accuracy:',clf.score(X_test,y_test))
