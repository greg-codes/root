# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:51:52 2019

@author: Tebe
"""

#%%
import pandas as pd
import numpy as np
import time, os, glob, sys
import datetime

import category_encoders as ce
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

sys.path.append(os.getcwd())
import load_file as lf
import model_metrics as mm
#%%
#Load ALL of the data
df = pd.DataFrame()
for f in tqdm(glob.glob(r'C:\Users\Tebe\Documents\Root Ad Data\csvs\*.gzip')):
#	print(f)
	df = pd.concat([df,lf.temp_load(fname=f)], axis=1)
#print('Loaded')
df = df.sample(frac=1)
#%%
#Process data in minibatches
chunksize=100000
model = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=8)
hash_encoder = ce.HashingEncoder(n_components=200)
scaler = RobustScaler()
for n in  tqdm(np.arange(1,2)):
	X = df.iloc[n*chunksize:(n+1)*chunksize]
	y = X.clicks.values.astype(int)
	X = X.drop(['clicks','bid_timestamp_utc', 'tz', 'spend', 'installs'], axis=1)
	X['bid_timestamp_local'] = X['bid_timestamp_local'].dt.hour
	X['bid_floor'] = np.transpose(scaler.fit_transform([X.bid_floor.values]))
	X = hash_encoder.fit_transform(X,y)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	sm = SMOTE(random_state=2)
	X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
	
	#model = LogisticRegression(solver='saga',n_jobs=8, penalty='l2', verbose=5,C=4)
	#model.partial_fit(X_train_res, y_train_res, classes=np.unique(y))
	model.fit(X_train_res, y_train_res)
#%%
#diplay metrics for model performance
print(model.score(X_test, y_test))
mm.model_report_card(model, X_train_res, y_train_res, X_test, y_test, normalize=True)





