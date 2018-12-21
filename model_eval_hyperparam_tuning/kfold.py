import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Loading the data
df=pd.read_csv('wdbc.data',header=None)
X=df.loc[:,2:].values
y=df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)

# Performing train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)

# Making pipeline
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))

kfold=StratifiedKFold(n_splits=10,random_state=1).split(X_train,y_train)
scores=[]
for k,(train,test) in enumerate(kfold):
    pipe_lr.fit(X_train[train],y_train[train])
    score=pipe_lr.score(X_train[test],y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f'%(k+1,np.bincount(y_train[train]),score))

print('CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

# Using k-fold cross-validation scorer in sklearn
scores=cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=-1)
print('\nCV accuracy scores: %s'%scores)
print('CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))
