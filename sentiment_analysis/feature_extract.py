# Transforming words into feature vectors
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count=CountVectorizer()
docs=np.array(['The sun is shining','The weather is sweet','The sun is shining and the weather is sweet'])
bag=count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

tfidf=TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(bag).toarray())
