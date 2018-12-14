import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce


df=pd.DataFrame([['green','M',10.1,'class1'],
                            ['red','L',13.5,'class2'],
                            ['blue','XL',15.3,'class1']])
df.columns=['color','size','price','classlabel']
#print(df)

size_mapping={'XL':3,'L':2,'M':1}
df['size']=df['size'].map(size_mapping)
#print(df)

inv_size_mapping={v:k for k,v in size_mapping.items()}
#df['size']=df['size'].map(inv_size_mapping)
#print(df)

class_mapping={label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
#print(class_mapping)
df['classlabel']=df['classlabel'].map(class_mapping)
#print(df)

#alternative approach
class_le=LabelEncoder()
y=class_le.fit_transform(df['classlabel'].values)
#df['classlabel']=y
#print(df)
#df['classlabel']=class_le.inverse_transform(y)
#print(df)

X=df[['color','size','price']].values
#ohe=OneHotEncoder(categorical_features=[0])
#print(ohe.fit_transform(X).toarray())
#le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
#print(le.fit_transform(X))

print(pd.get_dummies(df[['price','size','color']], drop_first=True))
