#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from six.moves import urllib
import os


# In[2]:


#function to read the data
def load_data():
    data_path = "C:/"    
    csv_path = os.path.join(data_path, "full_gen_data.csv")
    return(pd.read_csv(csv_path)) 


# In[3]:


data = load_data()
data.head()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


#check if there is Nan Values 

cleaned = data.dropna()
cleaned.info()


# In[7]:


#Check if there is dublicated values

dublicated_check = cleaned.duplicated()
dublicated_check.value_counts()


# In[8]:


#Hereinbefore, we reached to a conclusion that there is no Nan values nor dublicates


# In[9]:


#Adding extra features

data['profit'] = data['current_price'] - data['cost']
data['discount_ratio'] = 1 - data['ratio']


# In[10]:


# Removing unwanted features

# Function to remove unwanted features
remove = ['article', 'retailweek', 'customer_id', 'article.1', 'sizes']

def remove_unwanted_features(data, rem_Feature_list):
       
    for ele in rem_Feature_list:
        for attr in (list(data)): 
            if attr == ele:
                data = data.drop(attr, axis = 1)
    return data

dataa = remove_unwanted_features(data, remove)
dataa


# In[11]:


# Exploring Data
dataa.hist(bins = 70, figsize = (15, 12))


# In[12]:


# Zooming more on discount_ratio feature
dataa["discount_ratio"].hist(bins = 70)


# In[13]:


# There is a correlation between discount_ratio and the convergence rate(label)
from pandas.plotting import scatter_matrix

dataa.plot(kind = 'scatter', x = 'discount_ratio', y = 'label', alpha = 0.002)


# In[14]:


# There is a correlation between sales amount in respective week and the convergence rate(label)
dataa.plot(kind = 'scatter', x = 'sales', y = 'label', alpha = 0.002)


# In[15]:


#Looking at the correlation between the dependant feature and independand features

corr_matrix = dataa.corr()
corr_matrix


# In[16]:


# Deciding the categorical features
list_cat_attr = list(dataa[['style', 'gender','country', 'productgroup', 'category']])

# Deciding the numerical features
list_num_attr = list(dataa.drop(list_cat_attr, axis = 1))

# Showing both Categorical and Numerical attributes
print(' Categorical features: ', list_cat_attr,'\n', 'Numerical Features: ', list_num_attr)


# In[17]:


# Removing dependant feature before scaling others

data2 = dataa.drop('label', axis = 1)


# In[18]:


#standardizing numberical features 

# Function to encode categorical features
def num_scaller(data, features_list, label_feature):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    
    
    # Removing the target feature from features list as it has 0 and 1 classes
    # and doesn't need to be scalled
    features_list.remove(label_feature)
    
    
    # Scalling the numerical features
    for ele in list(features_list): 
        data[[ele]] = scaler.fit_transform(data[[ele]]).reshape(-1,1)
    
    return data

# Calling the function to scall numerical features excep for target feature
num_scalled = num_scaller(data2, list_num_attr, 'label')
num_scalled


# In[19]:


# Re-adding the dependant feature (label)

num_scalled['label'] = dataa['label'] 
num_scalled


# In[20]:


# Encoding categorical features

# Encoding using labelEncoder
from sklearn import preprocessing
num = preprocessing.LabelEncoder()

# Function to encode categorical features

def cat_encoder(data, cat_list):
    for every in cat_list:
        for feature in list(data):
            if every == feature:
                num.fit(np.unique(dataa[feature]))
                data[feature]=num.transform(data[feature]).astype('int')
    return data

cat_scaled_encoded = cat_encoder(num_scalled, list_cat_attr)
cat_scaled_encoded


# In[21]:


#Rank the features of the dataset with recursive feature elimination (RFE) method
#and Random Forest Classifier algorithm as its estimator

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


X = np.asarray(cat_scaled_encoded.drop(['label'], axis = 1))
y = np.asarray(cat_scaled_encoded['label'])

rfc = RandomForestClassifier(n_estimators=40)
rfe = RFE(rfc, 8)
rfe_fit = rfe.fit(X, y)

print("Num Features: %s" % (rfe_fit.n_features_))
print("Selected Features: %s" % (rfe_fit.support_))
print("Feature Ranking: %s" % (rfe_fit.ranking_))


# In[22]:


# Function to get indices of features selected as relevant to the lebel feature
def select_features():
    
    # From RFE get list of indices of relevant features
    rfe_result = list(rfe_fit.ranking_)
    indices = []
    # Iterating over all list to find most relevant features( ==1), and append its index
    for each in range(len(rfe_result)):
        if rfe_result[each] == 1:
            indices.append(each)
    
    # Copying from data most relevant features in X, y dataframes 
    xc = cat_scaled_encoded.iloc[:,indices]
    X = np.asarray(xc)
    y = np.asarray(cat_scaled_encoded['label'])    
    
    return X, y
X, y = select_features()  


# In[23]:


#Resample the data to have balanced classes

from imblearn.over_sampling import SMOTE

sm=SMOTE(ratio='auto', kind='regular')
X_sampled,y_sampled=sm.fit_sample(X,y)


# In[24]:


Sampled_no = len(y_sampled[y_sampled==0])
Sampled_yes = len(y_sampled[y_sampled==1])
print([Sampled_no],[Sampled_yes])


# In[25]:


#The data now are balanced with 86072 for each class of label feature


# In[26]:


#Spliting the data into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_sampled,y_sampled,test_size=0.3,random_state=0)


# In[27]:


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


# In[28]:


#implementing cross-validation for the 4 other algorithms 

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score


sto_grad_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
rand_for_clf = RandomForestClassifier(n_estimators=40)
dec_tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
sv_clf = SVC(kernel='rbf', gamma='auto')
log_reg = LogisticRegression(C=1, solver='lbfgs')

print(
    ' Cross Validation for SGD: ', cross_val_score(sto_grad_clf, X_train, y_train_1, cv=3, scoring="accuracy"),
    '\n','Cross Validation for Random Forest: ',cross_val_score(rand_for_clf, X_train, y_train_1, cv=3, scoring="accuracy"),
    '\n','Cross Validation for Decision Tree: ',cross_val_score(dec_tree_clf, X_train, y_train_1, cv=3, scoring="accuracy"),
    '\n','Cross Validation for Decision Tree: ',cross_val_score(sv_clf, X_train, y_train_1, cv=3, scoring="accuracy"),
    '\n','Cross Validation for Logistic Regression: ', cross_val_score(log_reg, X_train, y_train_1, cv=3, scoring="accuracy")
    )


# In[29]:


# Getting F1 Score for difference machine learning algorithms

def scorer(a, b, c, d, e):
    for every in (a, b, c, d, e):
        every.fit(X_train,y_train)
        print (every.__class__.__name__, 'F1 score =', f1_score(y_test,every.predict(X_test)))
        
scorer(sto_grad_clf, rand_for_clf, dec_tree_clf, sv_clf, log_reg)


# In[30]:


#looking at the precision and recall for the algorithms

from sklearn.metrics import classification_report

def algorithms_metrics(a, b, c, d, e):
    for every in (a, b, c, d, e):
        yhat = every.predict(X_test)
        print(every.__class__.__name__, '\n')
        print(classification_report(y_test,yhat))
    
algorithms_metrics(sto_grad_clf, rand_for_clf, dec_tree_clf, sv_clf, log_reg)


# In[31]:


#looking at the precision and recall for best algorithm

yhat = sv_clf.predict(X_test)
print(' Metrics for best algorithm which is Support Vector Classifier')
print(classification_report(y_test,yhat))


# In[32]:


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

plot_confusion_matrix(confusion_matrix(y_test, yhat), classes=['0','1'], normalize= False,  title='Confusion matrix for SVC')


# In[33]:


# The best recall(95%) is for Support Vector Classifier, this means that
# the marketing campaign will reach to 95% of its targeted customers which are 74% of the current space of customers
# This means that the marketing campaign budget will be minimized to 26% and with 95% Return Of Invenstment on Marketing. 
# And this is a trad-off matter, Assuming that the marketing cost is only a small portion of the total cost of the article

