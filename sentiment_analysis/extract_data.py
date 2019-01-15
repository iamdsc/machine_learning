# Assembling individual text documents into a single CSV file
import pyprind
import pandas as pd
import os
import numpy as np

# change basepath to dir of unzipped movie dataset
basepath='dataset/aclImdb'

labels={'pos':1, 'neg':0}

# adding progress bar 
pbar=pyprind.ProgBar(50000)

df=pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path=os.path.join(basepath,s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                txt=infile.read()
            df=df.append([[txt, labels[l]]],ignore_index=True)
            pbar.update()
df.columns=['review','sentiment']

# shuffling the dataframe and storing in CSV file
np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',index=False,encoding='utf-8')
