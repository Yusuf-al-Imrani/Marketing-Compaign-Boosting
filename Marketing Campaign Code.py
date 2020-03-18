#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from six.moves import urllib
import os


# In[32]:


#function to read the data
def load_data():
    data_path = "C:/"    
    csv_path = os.path.join(data_path, "full_gen_data.csv")
    return(pd.read_csv(csv_path))


# In[33]:


data = load_data()
data.head()


# In[34]:


data.describe()


# In[35]:


data.info()


# In[36]:


#check if there is Nan Values 

cleaned = data.dropna()
cleaned.info()


# In[37]:


#Check if there is dublicated values

dublicated_check = cleaned.duplicated()
dublicated_check.value_counts()


# In[38]:


#Hereinbefore, we reached to a conclusion that there is no Nan values nor dublicates


# In[39]:


#Adding extra feature

data['profit'] = data['current_price'] - data['cost']
data['discount_ratio'] = 1 - data['ratio']
#data = data.drop(['sizes'], axis = 1)


# In[40]:


#Looking at the correlation between the dependant feature and independand features

corr_matrix = data.corr()
corr_matrix


# In[41]:


#Encoding categorical features

from sklearn import preprocessing
num = preprocessing.LabelEncoder()

num.fit(['Germany', 'Austria', 'France'])
data['country']=num.transform(data['country']).astype('int')

num.fit(["SHOES","HARDWARE ACCESSORIES","SHORTS","SWEATSHIRTS"])
data['productgroup']=num.transform(data['productgroup']).astype('int')

num.fit(["TRAINING","FOOTBALL GENERIC","RUNNING","RELAX CASUAL","INDOOR","GOLF"])
data['category']=num.transform(data['category']).astype('int')

num.fit(["women","men","unisex", "kids"])
data['gender']=num.transform(data['gender']).astype('int')

num.fit(["regular","wide","slim"])
data['style']=num.transform(data['style']).astype('int')


# In[42]:


#standardizing numberical features 

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

data['sales'] = scaler.fit_transform(data[['sales']]).reshape(-1,1)
data['regular_price'] = scaler.fit_transform(data[['regular_price']]).reshape(-1,1)
data['current_price'] = scaler.fit_transform(data[['current_price']]).reshape(-1,1)
#ratio is already between 0-1
#data['ratio'] = scaler.fit_transform(data[['ratio']]).reshape(-1,1)
data['cost'] = scaler.fit_transform(data[['cost']]).reshape(-1,1)

data['rgb_r_main_col'] = scaler.fit_transform(data[['rgb_r_main_col']]).reshape(-1,1)
data['rgb_g_main_col'] = scaler.fit_transform(data[['rgb_g_main_col']]).reshape(-1,1)
data['rgb_b_main_col'] = scaler.fit_transform(data[['rgb_b_main_col']]).reshape(-1,1)
data['rgb_r_sec_col'] = scaler.fit_transform(data[['rgb_r_sec_col']]).reshape(-1,1)
data['rgb_g_sec_col'] = scaler.fit_transform(data[['rgb_g_sec_col']]).reshape(-1,1)
data['rgb_b_sec_col'] = scaler.fit_transform(data[['rgb_b_sec_col']]).reshape(-1,1)


data['profit'] = scaler.fit_transform(data[['profit']]).reshape(-1,1)
#discount_ratio is already between 0-1
#data['discount_ratio'] = scaler.fit_transform(data[['discount_ratio']]).reshape(-1,1)


# In[47]:





# In[49]:


#Rank the features of the dataset with recursive feature elimination (RFE) method
#and Random Forest Classifier algorithm as its estimator

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


X = np.asarray(data.drop(data[['article', 'retailweek', 'customer_id', 'article.1', 'sizes', 'label']], axis = 1))
y = np.asarray(data['label'])

rfc = RandomForestClassifier(n_estimators=40)
rfe = RFE(rfc, 15)
rfe_fit = rfe.fit(X, y)

print("Num Features: %s" % (rfe_fit.n_features_))
print("Selected Features: %s" % (rfe_fit.support_))
print("Feature Ranking: %s" % (rfe_fit.ranking_))


# In[52]:


list(data.drop(data[['article', 'retailweek', 'customer_id', 'article.1', 'sizes', 'label']], axis = 1))


# In[54]:


X = np.asarray(data[['country', 'sales', 'regular_price', 'current_price', 'ratio','promo1',
                     'category', 'cost', 'style',
                     'rgb_r_main_col','rgb_g_main_col', 'rgb_b_main_col',
                     'rgb_g_sec_col',
                     'profit', 'discount_ratio']])

y = np.asarray(data['label'])


# In[55]:


#Resample the data to overcome oversampling

from imblearn.over_sampling import SMOTE

sm=SMOTE(ratio='auto', kind='regular')
X_sampled,y_sampled=sm.fit_sample(X,y)


# In[56]:


Sampled_no = len(y_sampled[y_sampled==0])
Sampled_yes = len(y_sampled[y_sampled==1])
print([Sampled_no],[Sampled_yes])


# In[57]:


#The data now are balanced with 86072 for each class of label feature


# In[58]:


#Spliting the data into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_sampled,y_sampled,test_size=0.3,random_state=0)


# In[59]:


#implementing cross-validation for Stocastic Gradient Descent

from sklearn.linear_model import SGDClassifier

y_train_1 = (y_train==1)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_1)


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_1):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_1[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_1[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


# In[60]:


#implementing cross-validation for the 4 other algorithms 

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score


lr = LogisticRegression(C=1, solver='lbfgs')
clf = SVC(kernel='rbf', gamma='auto')
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
rfc = RandomForestClassifier(n_estimators=40)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)


print(' Cross Validation for SGD: ', cross_val_score(sgd_clf, X_train, y_train_1, cv=3, scoring="accuracy"),
      '\n','Cross Validation for Logistic Regression: ', cross_val_score(lr, X_train, y_train_1, cv=3, scoring="accuracy"),
      '\n','Cross Validation for Decision Tree: ',cross_val_score(dtree, X_train, y_train_1, cv=3, scoring="accuracy"),
      '\n','Cross Validation for Random Forest: ',cross_val_score(rfc, X_train, y_train_1, cv=3, scoring="accuracy"))


# In[61]:


#trying machine learning algorithms



lr = LogisticRegression(C=1, solver='lbfgs')
clf = SVC(kernel='rbf', gamma='auto')
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
rfc = RandomForestClassifier(n_estimators=40)

def scorer(i,j,k,l):
    for every in (i,j,k,l):
        every.fit(X_train,y_train)
        print (every.__class__.__name__, 'F1 score =', f1_score(y_test,every.predict(X_test)))
        
scorer (lr,clf,dtree,rfc)


# In[62]:


#looking at the precision and recall for the algorithms

from sklearn.metrics import classification_report

def algorithms_metrics(a, b, c, d):
    for every in (a, b, c, d):
        yhat = every.predict(X_test)
        print(every.__class__.__name__, '\n')
        print(classification_report(y_test,yhat))
    
algorithms_metrics(lr, clf, dtree, rfc)


# In[63]:


#looking at the precision and recall for best algorithm

yhat = rfc.predict(X_test)
print(classification_report(y_test,yhat))


# In[64]:


#Visualizing the precision and recall confusion matrix

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_matrix(y_test, yhat), classes=['0','1'], normalize= False,  title='Confusion matrix')


# In[65]:


#looking at the precision and recall for best recall

yhat = dtree.predict(X_test)
print(classification_report(y_test,yhat))


# In[66]:



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_matrix(y_test, yhat), classes=['0','1'], normalize= False,  title='Confusion matrix')


# In[67]:


# The best recall(96%) is for Descision Tree Classifier, this means that
# the marketing campaign will reach to 96% of its targeted customers which are 76% of customers who received the offers
# and this means also that 24% of the marketing campaign budget will to to untargetted customers. 
# And this is a trad-off matter, Assuming that the marketing cost is only a small portion of the total cost of the article
# hence assuring to reach max number of customers with 100% convergence is worth this extra 24% of the marketing budget 

