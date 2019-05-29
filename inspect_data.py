# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:16:31 2019

@author: Greg Smith
"""
#%% load packages
import os, sys, time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams['figure.dpi'] = 200 # fix high-dpi display scaling issues

sys.path.append(os.getcwd()) # add cwd to path

from zip_codes import ZC # zip code database
import load_file as lf # file i/o
import myplots as mp # my plotting functions

zc = ZC() # initialize zip code class
#%% load data
fname_27 = '2019-04-27.csv'
fname_28 = '2019-04-28.csv'

data_dir = r'C:\PythonBC\RootData'

#df_27 = lf.load_wrapper(fname=fname_27, data_dir=data_dir, nrows=50000)
#df_28 = lf.load_wrapper(fname=fname_28, data_dir=data_dir, nrows=50000)

#%% save pd.df to disk after we are done modifying it
#temp_save(df_27, os.path.join(data_dir,fname_27.split('.')[0]+'.gzip'))
#temp_save(df_28, os.path.join(data_dir,fname_28.split('.')[0]+'.gzip'))

# load pandas.df from disk
df_27 = lf.temp_load(os.path.join(data_dir,fname_27.split('.')[0]+'.gzip'))
df_28 = lf.temp_load(os.path.join(data_dir,fname_28.split('.')[0]+'.gzip'))

#%% visualize data
#look at data that resulted in a click

ax = mp.make_countplot(df_27, col='platform_device_make', count='clicks')
ax = mp.make_countplot(df_27,col='platform_bandwidth', count='clicks')
ax = mp.make_countplot(df_27,col='platform_carrier', count='clicks')
ax = mp.make_countplot(df_27,col='platform_device_screen_size', count='clicks')
ax = mp.make_countplot(df_27,col='creative_type', count='clicks')
ax = mp.make_countplot(df_27,col='hour', count='clicks', order=False); ax.set_xlabel('local hour')
ax = mp.make_countplot(df_27,col='tz', count='clicks'); ax.set_xlabel('time zone')
ax = mp.make_countplot(df_27,col='state', count='clicks')

ax = mp.utc_vs_local(df_27)

g = sns.FacetGrid(df_27, col="clicks")
g = g.map(sns.distplot, "hour", bins=list(np.arange(0,25)), norm_hist=True, label='local hour')

g = sns.FacetGrid(df_27, col="clicks")
g = g.map(sns.distplot, "hour", bins=list(np.arange(0,25)), norm_hist=True, label='local hour')


#%% first pass at making a model for clicks
'''
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm

enc = OneHotEncoder()

### choose which variables to model with
bandwidth_27 = pd.get_dummies(df_27['platform_bandwidth'])
#device_make_27 = pd.get_dummies(df_27['platform_device_make'])
X_27 = df_27['hour']
X_27 = pd.concat([X_27, bandwidth_27], axis=1)

y_27 = df_27.clicks

clf = svm.LinearSVC()
clf.fit(X_27, y_27) 
print(f'score (self-fitting): {clf.score(X_27,y_27)}')

print('this model does not converge...')
'''
#%% compare this fit to another day
'''
print(f' # of bandwidths the same?: {len(df_27.platform_bandwidth.unique()) == len(df_28.platform_bandwidth.unique())}') 
print(f' # of device_makes the same?: {len(df_27.platform_device_make.unique()) == len(df_28.platform_device_make.unique())}') 
      
len(df_27.platform_device_make.unique())
len(df_28.platform_device_make.unique())

print('df_28 has a different set of device_makes. need to account for this somehow...')
      
bandwidth_28 = pd.get_dummies(df_28['platform_bandwidth'])
#device_make_28 = pd.get_dummies(df_28['platform_device_make'])
X_28 = df_28['hour']
X_28 = pd.concat([X_28, bandwidth_28], axis=1)
y_28 = df_28.clicks

print(f'score (train on 27th, test on 28th) = {clf.score(X_28,y_28)}')

from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_28, y_28, scoring='recall_macro', cv=5) 
'''

#%% determine number of categories

df_cats = lf.temp_load( os.path.join(data_dir, 'category.gzip') )
cats = df_cats.category.unique().tolist()
cats = [x for x in cats if str(x) != 'nan'] # remove nans
allcats = []
for i in cats:
    allcats.append( i.split(','))
allcats = [item for sublist in allcats for item in sublist] # flatten list of lists
allcats = list(dict.fromkeys(allcats)) # remove duplicates
print(f'there are {len(allcats)} unique categories in this dataset')

for c in allcats:
    print(c)