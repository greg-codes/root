# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:55:56 2019

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

zc = ZC('') # initialize zip code class
data_dir = r'C:\PythonBC\RootData'

#%%

def unique_counter(data_dir=r'C:\PythonBC\RootData', col='category'):
	df = lf.temp_load( os.path.join(data_dir, col+'.gzip') )
	uniqs = df[col].unique().tolist()
	uniqs = [x for x in uniqs if str(x) != 'nan'] # remove nans
	if col == 'category':
		allcats = []
		for i in uniqs:
			#allcats.append( i.split(',')) # comma-separated
			allcats.append( i.split(' ')) # space-separated
		allcats = [item for sublist in allcats for item in sublist] # flatten list of lists
		allcats = list(dict.fromkeys(allcats)) # remove duplicates
		print(f'there are {len(allcats)} unique IDF categories in this dataset')
		
	if col == 'geo_zip':
		df['geo_len'] = df.geo_zip.apply(len)
		print(f'of the strings, there are {len(df.geo_len.unique())} unique lengths')
	print(f'there are {df[col].isnull().sum()} null rows')
	print(f'there are {len(uniqs)} unique elements in {col}')
	if col == 'platform_carrier':
		print(f'there are {df.platform_carrier[df.platform_carrier == "-1"].count()} -1 entries')
	return df

#%% check inventory_source
df = unique_counter(col='inventory_source')
# 4 unique inventory_source: MOPUB, GOOGLE_ADX, PUBMATIC, RUBICON

#%% check app_bundle
df = unique_counter(col='app_bundle')
# 1 unique app_bundle: 0xcab1a940441cfe29

#%% check category column
df = unique_counter(col='category')

#%% check inventory_interstitial column
df = unique_counter(col='inventory_interstitial')
#there are 2 unique elements in inventory_interstitial
#True, False

#%% check inventory_interstitial column
df = unique_counter(col='geo_zip')
#of the strings, there are 1 unique lengths
#there are 363136 null rows
#there are 15326 unique elements in geo_zip

#%% check platform_bandwidth
df = unique_counter(col='platform_bandwidth')
#there are 8 unique elements in platform_bandwidth
#CELL_4G, WIFI, CELL_UNKNOWN, UNKNOWN, NaN, CELL_3G, ETHERNET, CONNECTION_UNKNOWN, CELL_2G

#%% platform_carrier
df = unique_counter(col='platform_carrier')
print(df.platform_carrier.value_counts())

### before cleaning
#there are 20948278 null rows
#there are 17 unique elements in platform_carrier
#T-Mobile                4826790
#Verizon                 4336020
#AT&T                    3341758
#Sprint                  2425372
#U.S. Cellular            265629
#C-Spire Wireless          96853
#-1                        62408
#Viaero                    14472
#Cellular One               9760
#West Central               7423
#Pioneer Cellular           2942
#Bluegrass Cellular         1241
#Appalachian Wireless        698
#i wireless                  661
#Cincinnati Bell             478
#ETEX Wireless               190
#Claro                        15
#Name: platform_carrier, dtype: int64

### after cleaning with .apply()
#there are 200866 null rows
#there are 506 unique elements in platform_device_make
#there are 0 -1 entries
#AT&T
#Appalachian Wireless
#Sprint
#Verizon
#T-Mobile
#nan
#C-Spire Wireless
#U.S. Cellular
#West Central
#Pioneer Cellular
#Cellular One
#Viaero
#i wireless
#Bluegrass Cellular
#Cincinnati Bell
#ETEX Wireless
#Claro

#after cleaning with .replace()
#there are 21010686 null rows
#there are 16 unique elements in platform_carrier
#T-Mobile                4826790
#Verizon                 4336020
#AT&T                    3341758
#Sprint                  2425372
#U.S. Cellular            265629
#C-Spire Wireless          96853
#Viaero                    14472
#Cellular One               9760
#West Central               7423
#Pioneer Cellular           2942
#Bluegrass Cellular         1241
#Appalachian Wireless        698
#i wireless                  661
#Cincinnati Bell             478
#ETEX Wireless               190
#Claro                        15
#Name: platform_carrier, dtype: int64
#%% platform_device_make
df = unique_counter(col='platform_device_make')
### before cleaning
#there are 0 null rows
#there are 508 unique elements in platform_device_make

### after cleaning with .apply()
#there are 200866 null rows
#there are 506 unique elements in platform_device_make

### after cleaning with .replace()
#there are 200866 null rows
#there are 506 unique elements in platform_device_make

#%% platform_device_model
df = unique_counter(col='platform_device_model')
### before cleaning
#there are 0 null rows
#there are 5996 unique elements in platform_device_model

### after cleaning with .replace()
#there are 71582 null rows
#there are 5995 unique elements in platform_device_model

### after cleaning with .apply()
#there are 71582 null rows
#there are 5995 unique elements in platform_device_model

### after cleaning with .replace()

#%% platform_device_screen_size
df = unique_counter(col='platform_device_screen_size')
print(df.platform_device_screen_size.value_counts())
### original data
#there are 116811 null rows
#there are 5 unique elements in platform_device_screen_size
#XL         34103809
#L           2057395
#UNKNOWN       62408
#M               518
#S                47
#Name: platform_device_screen_size, dtype: int64

### after cleaning with .apply( platform_dev_screen_cleaner )
#there are 62408 null rows
#there are 4 unique elements in platform_device_screen_size
#XL    34115339
#L      2057395
#S        64102
#M        41744
#Name: platform_device_screen_size, dtype: int64

### after cleaning with .replace()
#there are 179219 null rows
#there are 4 unique elements in platform_device_screen_size
#XL    34103809
#L      2057395
#M          518
#S           47
#Name: platform_device_screen_size, dtype: int64
#%% rewarded
df = unique_counter(col='rewarded')
print(df.rewarded.value_counts())

#there are 0 null rows
#there are 2 unique elements in rewarded
#False    34387971
#True      1953017
#Name: rewarded, dtype: int64
#%% bid_floor
df = unique_counter(col='bid_floor')
#there are 0 null rows
#there are 1487 unique elements in bid_floor
#          bid_floor
#count  3.634099e+07
#mean   2.633630e-03
#std    2.728339e-03
#min    0.000000e+00
#25%    9.000000e-05
#50%    1.870000e-03
#75%    4.210000e-03
#max    3.362000e-02

#%% bid_timestamp_utc
df = unique_counter(col='bid_timestamp_utc')
# i stopped running this after about 3 minutes

#%% day_of_week
df = unique_counter(col='day_of_week')
print(df.day_of_week.value_counts())
#there are 0 null rows
#there are 7 unique elements in day_of_week
#thursday     6733765
#friday       6312794
#saturday     5459572
#wednesday    5173277
#tuesday      4560872
#sunday       4451961
#monday       3648747
#Name: day_of_week, dtype: int64


#%% segments
df = unique_counter(col='segments')

segs = df.segments.unique().tolist()
segs = [x for x in segs if str(x) != 'nan'] # remove nans
allsegs = []
for i in segs:
	allsegs.append( i.split(',')) # comma-separated
	#allsegs.append( i.split(' ')) # space-separated
allsegs = [item for sublist in allsegs for item in sublist] # flatten list of lists
allsegs = list(dict.fromkeys(allsegs)) # remove duplicates
print(f'there are {len(segs)} unique bundles of segments in this dataset')
print(f'there are {len(allsegs)} unique segments in this dataset')

### before cleaning
#there are 0 null rows
#there are 441086 unique elements in segments
#segments looks similar to category, list of strings. need to split, clean, sort, concat.
#there are 441086 unique bundles of segments in this dataset
#there are 199 unique segments in this dataset
#this list of "unique segments" contains duplicates

### after cleaning with apply( segment_cleaner )
#there are 425523 unique bundles of segments in this dataset
#there are 51 unique segments in this dataset
#this list of "unique segments" contains duplicates

#%% creative_type
df = unique_counter(col='creative_type')
print(df.creative_type.value_counts())

#%% creative_size
df = unique_counter(col='creative_type')
df2 = unique_counter(col='creative_size')
print(df.creative_size.value_counts())
df3 = pd.concat([df,df2], axis=1)
print(df3[df3.creative_size == '320x480'].creative_type.value_counts())
print(df3[df3.creative_size == '0x0'].creative_type.value_counts())
### note: creative_size and creative_type are redundant
# 320x480 == banner
# 0x0 == video
# for this reason, we will drop the creative_size column

#%% spend
df = unique_counter(col='spend')
#there are 0 null rows
#there are 4690 unique elements in spend
# interesting that there are less than 4700 unique spends

#%% clicks
df = unique_counter(col='clicks')
#there are 0 null rows
#there are 2 unique elements in clicks

#%% installs
df = unique_counter(col='installs')
#there are 0 null rows
#there are 2 unique elements in installs

#%% tz
df = unique_counter(col='tz')
#there are 368136 null rows
#there are 20 unique elements in tz

#%% local_ordinals.gzip
df = lf.temp_load( os.path.join(data_dir,'local_ordinals.gzip'))
print(df.hour.value_counts())
print(df.day.value_counts())
print(df.day_of_week.value_counts())

#%% bid_timestamp_local.gzip
df = lf.temp_load( os.path.join(data_dir,'bid_timestamp_local.gzip'))
print(f'there are {len(df[df.bid_timestamp_local < pd.Timestamp(2005, 1, 1, 12)])} bad timestamps')
