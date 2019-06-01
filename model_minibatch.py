# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:51:52 2019

@author: Tebe
"""

#%%
import pandas as pd
import numpy as np
import time, os, glob, sys

import category_encoders as ce
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import recall_score

sys.path.append(os.getcwd())
import load_file as lf
import model_metrics as mm

#%%
#Load all gzip data into large dataframe
data_directory = r'D:\Root Data\csvs'
#Load ALL of the data
df = pd.DataFrame()
for f in tqdm(glob.glob(os.path.join(data_directory, '*.gzip'))):
	df = pd.concat([df,lf.temp_load(fname=f)], axis=1)
df = df.sample(frac=1)
df = df.drop(['app_bundle','bid_timestamp_utc', 'tz', 'spend', 'installs', 'bid_timestamp_local'], axis=1)
df['inventory_interstitial'] = df['inventory_interstitial'].astype(int)
df['rewarded'] = df['rewarded'].astype(int)
df['clicks'] = df['clicks'].astype(int)
#%%
##Process data in minibatches
#chunksize=100000
#model = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1)
#hash_encoder = ce.HashingEncoder(n_components=2000)
#loo = ce.LeaveOneOutEncoder()
#scaler = StandardScaler()
#for n in  tqdm(np.arange(1,3)):
#	X = df.iloc[n*chunksize:(n+1)*chunksize]
#	y = X.clicks.values
#	X = X.drop(['clicks'], axis=1)
#	X = loo.fit_transform(X,y)
#	
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#	sm = SMOTEENN()
#	X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
#	
#	model.partial_fit(X_train_res, y_train_res, classes=np.unique(y))
##	model.fit(X_train_res, y_train_res)
##%%
##diplay metrics for model performance
#print(model.score(X_test, y_test))
#mm.model_report_card(model, X_train_res, y_train_res, X_test, y_test, normalize=False)
#%%
#pca = PCA(n_components = 0.9)
#X_pca = pca.fit_transform(X_train_res)
#print(X_pca.shape)
##fig, ax = plt.subplots()
##ax.scatter(X_pca[:,0],X_pca[:,1], c=y_train_res)
#%%
#Let's try the simple minded way
y = df.clicks.values
X = df.drop(['clicks'], axis=1)
loo = ce.LeaveOneOutEncoder()
print('Starting LOO encoding')
X = loo.fit_transform(X,y)
print('Done encoding')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#%%
#Process data in minibatches
chunksize=200000
model = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.001,n_jobs=-1)
sm = SMOTEENN(sampling_strategy = 'minority')
print('Starting to train...')
for n in  tqdm(np.arange(1,127)):
	X_train_chunky = X_train[n*chunksize:(n+1)*chunksize]
	y_train_chunky = y_train[n*chunksize:(n+1)*chunksize]
	X_train_chunky = loo.fit_transform(X_train_chunky,y_train_chunky)
	X_train_res, y_train_res = sm.fit_sample(X_train_chunky, y_train_chunky)
	model.partial_fit(X_train_res, y_train_res, classes=np.unique(y))
dump(model, 'SGD_model_minibatch_200k_127.joblib')
#%%
#diplay metrics for model performance
y_pred =  model.predict(X_test)
acc_score = model.score(X_test, y_test)
print(f'The accuracy score is: {acc_score}')
acc_score = recall_score(y_test, y_pred)
print(f'The accuracy score is: {acc_score}')
#%%
#mm.model_report_card(model, X_train_res, y_train_res, X_test, y_test, normalize=False)
fig, ax = plt.subplots()
mm.plot_confusion_matrix(y_test, y_pred, ax=ax, normalize=True)
fig, ax = plt.subplots()
mm.plot_confusion_matrix(y_test, y_pred, ax=ax, normalize=False)

#%%
#pca = PCA(n_components = 0.9)
#X_pca = pca.fit_transform(X_train)
#print(X_pca.shape)
##fig, ax = plt.subplots()
##ax.scatter(X_pca[:,0],X_pca[:,1], c=y_train_res)
#%%
#fn = 'df_full.gzip'
#lf.temp_save(fname=os.path.join(data_directory,fn), df=df)










