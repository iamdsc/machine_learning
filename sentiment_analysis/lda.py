import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df=pd.read_csv('movie_data.csv',encoding='utf-8')
#print(df.head())
count=CountVectorizer(stop_words='english',max_df=0.1,max_features=5000)
X=count.fit_transform(df['review'].values)

# fit LDA estimator to bag-of-words matrix and infer 10 different topics
lda=LatentDirichletAllocation(n_topics=10,random_state=123,learning_method='online')
X_topics=lda.fit_transform(X)

# lda.components_ stores a matrix containing the word importance for each of the topics
# in increasing order
# retrieving 5 most important words from each of the topics
n_top_words=5
feature_names=count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx+1))
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))

# plotting 3 movies from horror movie category
horror=X_topics[:,6].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx+1))
    print(df['review'][movie_idx][:300],'...')
