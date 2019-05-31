# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:45:06 2019

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

import seaborn as sns
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from scipy.sparse import hstack, csr_matrix 
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
#from yellowbrick.classifier import ClassificationReport
from sklearn.utils.multiclass import unique_labels

from tqdm import tqdm

sys.path.append(os.getcwd())
import load_file as lf
import model_metrics as mm

plt.rcParams['figure.dpi'] = 240
#%%
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
#geo_zip = vectorizer.fit_transform(df_balanced.geo_zip.apply(str))
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression(solver='saga',n_jobs=8, penalty='l2', verbose=5,C=0.01)
model.fit(X_train, y_train)
mm.model_report_card(model, X_train, y_train, X_test, y_test)









