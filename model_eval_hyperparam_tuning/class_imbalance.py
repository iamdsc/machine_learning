import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# Loading the data
df=pd.read_csv('wdbc.data',header=None)
X=df.loc[:,2:].values
y=df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)

print(np.unique(y,return_counts=True))

X_imb = np.vstack((X[y==0],X[y==1]))
y_imb = np.hstack((y[y==0],y[y==1]))

print('Number of class 1 samples before:',X_imb[y_imb==1].shape[0])
X_upsampled,y_upsampled=resample(X_imb[y_imb==1],y_imb[y_imb==1],replace=True,n_samples=X_imb[y_imb==0].shape[0],random_state=123)
print('Number of class 1 samples after:',X_upsampled[y_upsampled==1].shape[0])

# obtaining a balanced dataset
X_bal=np.vstack((X[y==0],X_upsampled))
y_bal=np.hstack((y[y==0],y_upsampled))
