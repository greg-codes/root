# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:33:55 2019

@author: Tebe
"""

#%%
import sys, os, glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import time

from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from scipy.sparse import hstack, csr_matrix 
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score

sys.path.append(os.getcwd())
import load_file as lf

plt.rcParams['figure.dpi'] = 240
#%%
def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure()
    plt.title('Logistic Regression ROC Curve', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This is a function that prints and plots the confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
directory = 'C:/Users/Tebe/Documents/Root Ad Data/csvs'
fname = '2019-04-20.csv'
print(fname)

# Load files from several days
df = lf.load_data(fname=fname, data_dir=directory)
for i in np.arange(1,2):
	fn = fname.split('0.')[0]+str(i)+'.csv'
	print(fn)
	df = df.append(lf.load_data(fname=fn, data_dir=directory))

#%%
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='UNKNOWN')
imp.fit(df[['platform_device_screen_size']])
df['platform_device_screen_size'] = imp.transform(df[['platform_device_screen_size']])
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='-1')
imp.fit(df[['platform_carrier']])
df['platform_carrier'] = imp.transform(df[['platform_carrier']])
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='UNKNOWN')
imp.fit(df[['platform_bandwidth']])
df['platform_bandwidth'] = imp.transform(df[['platform_bandwidth']])

#%%
#Implement Random-Under sampling

#First, shuffle dataframe
df = df.sample(frac=1)

#Create a balanced dataset
number_of_clicks = len(df.loc[df['clicks'] == 1])

df_clicks = df.loc[df['clicks'] == 1]
df_non_clicks = df.loc[df['clicks'] == 0][:number_of_clicks]
df_balanced = pd.concat([df_clicks, df_non_clicks])
#%%
#Encoding categorical data using the "hashing trick"

vectorizer = FeatureHasher(n_features=2**25, input_type='string')
invent_src = vectorizer.fit_transform(df_balanced.inventory_source)
#geo_zip = vectorizer.fit_transform(df_balanced.geo_zip)
screen_size = vectorizer.fit_transform(df_balanced.platform_device_screen_size)
carrier = vectorizer.fit_transform(df_balanced.platform_carrier)
bandwidth = vectorizer.fit_transform(df_balanced.platform_bandwidth)
maker = vectorizer.fit_transform(df_balanced.platform_device_make)
model = vectorizer.fit_transform(df_balanced.platform_device_model)
day_of_week = vectorizer.fit_transform(df_balanced.day_of_week)
scaler = RobustScaler()#StandardScaler()
bid_floor = np.transpose(csr_matrix(scaler.fit_transform([df_balanced.bid_floor.values])))
#spend = np.transpose(csr_matrix(scaler.fit_transform([df_balanced.spend.values])))

#%%
y = df_balanced['clicks']
X = hstack([invent_src, screen_size, carrier, bandwidth, maker, model, day_of_week, bid_floor])
#%%
#Dimensionality reduction using TruncatedSVD
'''
# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=4, algorithm='randomized').fit_transform(X)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))
'''
#%%
## TruncatedSVD scatter plot, only for n_components=2
#fig, ax = plt.subplots()
#ax.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Click', linewidths=2, alpha=0.1)
#ax.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Click', linewidths=2, alpha=0.1)
#ax.set_title('Truncated SVD', fontsize=14)

#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#%%
'''
#SMOTE implementation, will break your machine

vectorizer = FeatureHasher(n_features=2**22, input_type='string')
invent_src = vectorizer.fit_transform(df.inventory_source)
#geo_zip = vectorizer.fit_transform(df_balanced.geo_zip)
screen_size = vectorizer.fit_transform(df.platform_device_screen_size)
carrier = vectorizer.fit_transform(df.platform_carrier)
bandwidth = vectorizer.fit_transform(df.platform_bandwidth)
maker = vectorizer.fit_transform(df.platform_device_make)
model = vectorizer.fit_transform(df.platform_device_model)
day_of_week = vectorizer.fit_transform(df.day_of_week)
bid_floor = np.transpose(csr_matrix(df.bid_floor.values))
spend = np.transpose(csr_matrix(df.spend.values))

y = df['clicks']
#,'platform_device_screen_size
X = hstack([invent_src, screen_size, carrier, bandwidth, maker, model, day_of_week, bid_floor, spend])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('CTR before SMOTE:',100.0*sum(y_train)/len(y_train),'% ', \
	  'CTR after SMOTE:',100.0*sum(y_train_res)/len(y_train_res),'%')
'''
#%%
model = LogisticRegression(solver='saga',n_jobs=8, penalty='l2', verbose=5,C=0.01)
model.fit(X_train, y_train)
print('Model score: ', model.score(X_train,y_train))

#%%
'''
model = SGDClassifier(loss='log', n_jobs=8)
model.fit(X_train, y_train)
#model.fit(X_train, y_train)
print(model.score(X_train,y_train))
'''
#%%
'''
training_score = cross_val_score(model, X_train, y_train, cv=5)
print(training_score)
'''
#%%
'''
# GridSearchCV to optimize parameters in LogisticRegression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.1, 1, 10]}#[0.001, 0.01, 0.1, 1, 10, 100, 1000]

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, n_jobs=4)
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_
'''
#%%
y_train_pre = model.predict(X_train)
cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)
con_mat = confusion_matrix(y_train, y_train_pre)
print(con_mat)
#plot_confusion_matrix(con_mat)
#%%
#Plot ROC curve to get an idea of performance
precision, recall, threshold = precision_recall_curve(y_train, y_train_pre)
fpr, tpr, thresold = roc_curve(y_train, y_train_pre)   
logistic_roc_curve(fpr, tpr)
plt.show()
















