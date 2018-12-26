# Plotting a Receiver Operating Characteristic (ROC) curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc
from scipy import interp

# Loading the data
df=pd.read_csv('wdbc.data',header=None)
X=df.loc[:,2:].values
y=df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)

# Performing train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)

# Making pipeline
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(penalty='l2',random_state=1,C=100))

X_train2=X_train[:,[4,14]]
cv=list(StratifiedKFold(n_splits=3,random_state=1).split(X_train,y_train))

fig=plt.figure(figsize=(7,5))

mean_tpr=0
mean_fpr=np.linspace(0, 1, 100)
all_tpr=[]

for i, (train,test) in enumerate(cv):
    probas=pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test],probas[:,1],pos_label=1)
    # interploating the average ROC curve
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,tpr,label='ROC fold %d (area = %0.2f)' % (i+1,roc_auc))

plt.plot([0,1],[0,1],linestyle='--',color=(0.6,0.6,0.6),label='random guessing')

mean_tpr/=len(cv)
mean_tpr[-1]=1
mean_auc=auc(mean_fpr,mean_tpr)
plt.plot(mean_fpr,mean_tpr,'k--',label='mean ROC (area = %0.2f)'%mean_auc, lw=2)
plt.plot([0,0,1],[0,1,1],linestyle=':',color='black',label='perfect performance')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')
plt.show()
