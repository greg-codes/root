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
plt.rcParams['figure.dpi'] = 240 # fix high-dpi display scaling issues

sys.path.append(os.getcwd()) # add cwd to path

from zip_codes import ZC # zip code database
import load_file as lf # file i/o
import myplots as mp # my plotting functions

#%% load data

fname_01 = '2019-04-01.csv'
fname_27 = '2019-04-27.csv'
fname_28 = '2019-04-28.csv'

data_dir = r'C:\PythonBC\RootData'
zc = ZC(fdir='') # initialize zip code class

#df_01 = lf.load_wrapper(fname=fname_01, data_dir=data_dir)
#df_27 = lf.load_wrapper(fname=fname_27, data_dir=data_dir)
#df_28 = lf.load_wrapper(fname=fname_28, data_dir=data_dir, nrows=50000)

#%% save pd.df to disk after we are done modifying it
#temp_save(df_27, os.path.join(data_dir,fname_27.split('.')[0]+'.gzip'))
#temp_save(df_28, os.path.join(data_dir,fname_28.split('.')[0]+'.gzip'))

# load pandas.df from disk
#df_27 = lf.temp_load(os.path.join(data_dir,fname_27.split('.')[0]+'.gzip'))
#df_28 = lf.temp_load(os.path.join(data_dir,fname_28.split('.')[0]+'.gzip'))

#%% visualize data
#look at data that resulted in a click
'''
ax = mp.make_countplot(df_27, col='platform_device_make', count='clicks')
ax = mp.make_countplot(df_27,col='platform_bandwidth', count='clicks')
ax = mp.make_countplot(df_27,col='platform_carrier', count='clicks')
ax = mp.make_countplot(df_27,col='platform_device_screen_size', count='clicks')
ax = mp.make_countplot(df_27,col='creative_type', count='clicks')
ax = mp.make_countplot(df_27,col='hour', count='clicks', order=False); ax.set_xlabel('local hour')
ax = mp.make_countplot(df_27,col='tz', count='clicks'); ax.set_xlabel('time zone')
ax = mp.make_countplot(df_27,col='inventory_source', count='clicks')

ax = mp.utc_vs_local(df_27)

g = sns.FacetGrid(df_27, col="clicks")
g = g.map(sns.distplot, "hour", bins=list(np.arange(0,25)), norm_hist=True, label='local hour')

g = sns.FacetGrid(df_27, col="clicks")
g = g.map(sns.distplot, "hour", bins=list(np.arange(0,25)), norm_hist=True, label='local hour')
'''
#%% histograms using .gzip

#df = lf.temp_load( os.path.join(data_dir, 'day_of_week.gzip' ))
#df2 = lf.temp_load( os.path.join(data_dir, 'clicks.gzip' ))
#df3 = pd.concat( [df,df2], axis=1)
#ax = mp.make_countplot(df3,col='day_of_week', count=None, order=False); ax.set_xlabel('day_of_week')


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

#df_cats = lf.temp_load( os.path.join(data_dir, 'category.gzip') )
#cats = df_cats.category.unique().tolist()
#cats = [x for x in cats if str(x) != 'nan'] # remove nans
#allcats = []
#for i in cats:
#	#allcats.append( i.split(',')) # comma-separated
#	allcats.append( i.split(' ')) # space-separated
#allcats = [item for sublist in allcats for item in sublist] # flatten list of lists
#allcats = list(dict.fromkeys(allcats)) # remove duplicates
#print(f'there are {len(cats)} unique bundles of categories in this dataset')
#print(f'there are {len(allcats)} unique IDF categories in this dataset')

#for c in allcats:
#	print(c)

#%% verify that all dataframes have the same shape
'''
import glob
gzips = glob.glob( os.path.join(data_dir, '*.gzip') )[2:] # list of column gzips
name=[]; mem=[]; shape=[]; uniques=[]
for f in gzips:
	name.append( f.split('\\')[-1].split('.gzip')[0] )
	print(f'loading {name[-1]}...', end='')
	df = lf.temp_load( f )
	mem.append( lf.mem_usage(df) )
	shape.append( df_cats.shape )
	uniques.append( len(df.iloc[:,0].unique()) ) # unique elements
	print(' done')

mem2 = [ float(s.split(' MB')[0]) for s in mem ] # convert mem to floats

results = pd.DataFrame(np.column_stack([name, mem2, shape, uniques]), columns=['name', 'mem MB', 'rows', 'cols', 'uniques'])
results['mem MB'] = results['mem MB'].astype(float)
print(results)
print(f'total memory footprint = {results["mem MB"].sum():5.2f} MB')
'''
#%% look at histograms
'''
fname = 'platform_carrier.gzip'
df_clicks = lf.temp_load(os.path.join(data_dir,'clicks.gzip'))
df_var = lf.temp_load(os.path.join(data_dir,fname))

df = pd.concat([df_var, df_clicks], axis=1)

ax = mp.make_countplot(df,col='platform_carrier', count='clicks')
'''
#%% zipcode-level analysis
'''
df1 = lf.temp_load(os.path.join(data_dir,'geo_zip.gzip'))
df2 = lf.temp_load(os.path.join(data_dir,'clicks.gzip'))
frames = [df1, df2]
df = pd.concat(frames, axis=1)
print(f'there are { df.geo_zip.isnull().sum()} ({100*df.geo_zip.isnull().sum() / df.shape[0]:3.3f}%) null zip codes ')

df['zip_valid'] = df.geo_zip.notnull()
# question: is there any correlation between a valid zip code and click status?
ax = mp.make_countplot(df,col='zip_valid', count='clicks')

df = df.dropna(subset=['geo_zip']) # drop NaN values of zip codes
zc = ZC()
df['state'] = zc.zip_to_state_2(df.geo_zip)
print(f'there are { df.state.isnull().sum()} ({100*df.state.isnull().sum() / df.shape[0]:3.3f}%) null state rows')
'''
#%% state-level analysis
'''
df1 = lf.temp_load(os.path.join(data_dir,'geo_zip.gzip'))
df2 = lf.temp_load(os.path.join(data_dir,'clicks.gzip'))
frames = [df1, df2]
df = pd.concat(frames, axis=1)
zc = ZC()
df['state'] = zc.zip_to_state_2(df.geo_zip)
ax = mp.make_countplot(df,col='state', count='clicks')
'''
#%% day of week analysis
'''
df1 = lf.temp_load(os.path.join(data_dir,'day_of_week.gzip'))
df2 = lf.temp_load(os.path.join(data_dir,'clicks.gzip'))
frames = [df1, df2]
df = pd.concat(frames, axis=1)
ax = mp.make_countplot(df,col='day_of_week', count='clicks')


'''

#%% local hour analysis
'''
df1 = lf.temp_load(os.path.join(data_dir,'geo_zip.gzip'))
df2 = lf.temp_load(os.path.join(data_dir,'clicks.gzip'))
df3 = lf.temp_load(os.path.join(data_dir,'bid_timestamp_utc.gzip'))
frames = [df1, df2, df3]
df = pd.concat(frames, axis=1)
df = df.dropna(subset=['geo_zip']) # drop NaN values of zip codes
zc = ZC()
df['tz'] = zc.zip_to_tz_2(df.geo_zip)
df['tz'] = df.tz.astype('category')

df['bid_timestamp_local'] = zc.shift_tz_wrap(df) # compute local time
df['hour'] = zc.local_hour(df) # compute local hour
'''