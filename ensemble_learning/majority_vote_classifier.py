# majority vote classifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.externals import six
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ A majority vote ensemble classifier """
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers=classifiers
        self.named_classifiers={key:value for key,value in _name_estimators(classifiers)}
        self.vote=vote
        self.weights=weights

    def fit(self, X, y):
        """ Fit classifiers """
        # Use LabelEncoder to ensure class labels start with 0
        self.lablenc_=LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_=self.lablenc_.classes_
        self.classifiers_=[]
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ predict class labels for X """
        if self.vote == 'probability':
            maj_vote=np.argmax(self.predict_proba(X), axis=1)
        else:
            # collect results from clf.predict calls
            predictions=np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote=np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)

        maj_vote=self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba=np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for Grid Search """
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out=self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s'%(name,key)]=value
            return out
    
iris = datasets.load_iris()
X, y=iris.data[50:,[1,2]],iris.target[50:]
le=LabelEncoder()
y=le.fit_transform(y)

# performing train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

clf1=LogisticRegression(penalty='l2',C=0.001, random_state=1)
clf2=DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3=KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1=Pipeline([['sc',StandardScaler()],['clf',clf1]])
pipe3=Pipeline([['sc',StandardScaler()],['clf',clf3]])

clf_labels=['Logistic regression', 'Decision Tree', 'KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores=cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print('ROC AUC: %0.2f (+/- %0.2f) [%s]'%(scores.mean(),scores.std(),label))

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels+=['Majority voting']
all_clf=[pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores=cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print('Accuracy: %0.2f (+/- %0.2f)[%s]'%(scores.mean(), scores.std(), label))
