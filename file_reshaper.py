# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:17:01 2019

@author: Greg Smith

"""

import os, glob, sys
sys.path.append(os.getcwd()) # add cwd to path
import load_file as lf
import pandas as pd
from zip_codes import ZC # zip code database
#import numpy as np
#from zipfile import ZipFile

def reshape_files(data_dir=r'C:\PythonBC\RootData', f_ext='.csv', big_zip=False, all_cols=True, sub_cols=['geo_zip'], **kwargs):
	'''
	'reshapes' the data files. the original format of the data files provided
	by root gives us one day's worth of data per file, with all variables
	present. this function rearranges the format of the files so that each file
	contains data from all days, but only one column. files are saved via the
	parquet method (non-human-readable) as <column>.gzip for each column.
	
	running this file will 'reshape' all files in data_dir matching *.f_ext.
	
	input parameters:
		data_dir: location of files
		f_ext: filename extension ('.csv' or '.zip') of files to be loaded. 
			Ignored if big_zip == True.
		big_zip: True or False. If False, will perform the reshaping on all 
			files in data_dir matching *f_ext. If True, the function will
			perform the reshaping on bigzipname (hard-coded).
		all_cols: set True to load all columns from fname. set False if you 
			only want a subset.
		sub_cols: only implemented if all_cols==False. sub_cols is a list of 
			strings of column names to be loaded. must be comprised of the 
			elements of the mycols list (defined below). note: sub_cols 
			overrides the hard-coded <unwanted> list.
	'''
	# define the file name of the large zip file. assumed structure is a .zip
	# file with many .csv files inside
	bigzipname = 'root_ad_auction_dataset_all_data.zip'

	### determine which columns to load
	# list of columns in the file
	mycols = ['',
			  'auction_id',
			  'inventory_source',
			  'app_bundle',
			  'category',
			  'inventory_interstitial',
			  'geo_zip',
			  'platform_bandwidth',
			  'platform_carrier',
			  'platform_os',
			  'platform_device_make',
			  'platform_device_model',
			  'platform_device_screen_size',
			  'rewarded',
			  'bid_floor',
			  'bid_timestamp_utc',
			  'hour',
			  'day',
			  'day_of_week',
			  'month',
			  'year',
			  'segments',
			  'creative_type',
			  'creative_size',
			  'spend',
			  'clicks',
			  'installs']

	# always drop these columns
	unwanted = ['auction_id', 'platform_os', 'day', 'month', 'year', 'hour', 'creative_size', 'day_of_week']
	
	#drop the unwanted columns from mycols
	mycols = [ele for ele in mycols if ele not in unwanted]

	if not all_cols:
		# extract the user-defined columns from <mycols> if they are also in <sub_cols>
		mycols = [ele for ele in mycols if ele in sub_cols]
	
	### determine which files to load
	# get a list of all the csv files in the data director
	myfiles = glob.glob( os.path.join(data_dir, '*'+f_ext) )

	# sort <myfiles>
	myfiles = sorted( myfiles )
	
	# remove the big zip file from <myfiles>
	if bigzipname in myfiles:
		myfiles.remove(bigzipname)
	
	### load the columns and files
	
	if big_zip: # load the files from the mega zip file
		print(f"this code not written yet. let's pretend this opened {bigzipname}")
	else: # load files from individual csv/zip files
		if all_cols: # special case: need to remove the leading '' column
			for col in mycols[1:]: # loop through all items in mycols
				print(f'loading {col}...', end='')
				df_from_each_file = (lf.load_data(data_dir=data_dir, fname=f, all_cols=False, sub_cols=[col], **kwargs) for f in myfiles)
				df = pd.concat(df_from_each_file, ignore_index=True)
				print(' saving ...', end='')
				myfname = col + '.gzip'
				lf.temp_save(df, os.path.join(data_dir, myfname) ) # save to disk using parquet method
				print(f' {myfname} saved')
		else:
			for col in mycols: # loop through all items in mycols
				print(f'loading {col}...', end='')
				df_from_each_file = (lf.load_data(data_dir=data_dir, fname=f, all_cols=False, sub_cols=[col], **kwargs) for f in myfiles)
				df = pd.concat(df_from_each_file, ignore_index=True)
				print(' saving ...', end='')
				myfname = col + '.gzip'
				lf.temp_save(df, os.path.join(data_dir, myfname) ) # save to disk using parquet method
				print(f' {myfname} saved')
	print('all done!')
	return

def local_hour_creator(data_dir=r'C:\PythonBC\RootData', f_ext='.csv', big_zip=False, **kwargs):
	'''
	- calculates the bid_timestamp_local for all CSV files and saves to gzip.
	- also calculates the local hour, local day, and local day_of_the_week and
		saves to gzip
	- converting to local time is a computationally slow/expensive processes, 
		so we load one file at a time, and save those results to a temporary
		file. once all times in all files have been calculated, we merge the 
		results into a pair of files.
	saves two files:
		local_ordinals.gzip: hour, day and day_of_week (int8)
		bid_timestamp_local.gzip: timestamps
	
	'''
	# define the file name of the large zip file. assumed structure is a .zip
	# file with many .csv files inside
	bigzipname = 'root_ad_auction_dataset_all_data.zip'
	
	#initalize zipcode class
	zc = ZC('')
	
	#get sorted list of files to loop over
	myfiles = sorted( glob.glob( os.path.join(data_dir, '*'+f_ext) ) )

	# remove the big zip file from <myfiles>
	if bigzipname in myfiles:
		myfiles.remove(bigzipname)
	if os.path.join(data_dir,'2019-04-00.csv') in myfiles: # this file is empty
		myfiles.remove( os.path.join(data_dir,'2019-04-00.csv') )

	if big_zip: # load the files from the mega zip file
		print(f"this code not written yet. let's pretend this opened {bigzipname}")
	else: # load files from individual csv/zip files
		flist_TS_local = []
		flist_locals = []
		flist_tz = []
		flist_state = []
		for f in myfiles:
			print(f'loading {os.path.basename(f)} ... ', end='')
			# load UTC timestamp and zip code info from CSV files
			df_TS_utc = lf.load_data(data_dir=data_dir, fname=f, all_cols=False, sub_cols=['bid_timestamp_utc'], **kwargs)
			df_geozip = lf.load_data(data_dir=data_dir, fname=f, all_cols=False, sub_cols=['geo_zip'], **kwargs)
			df = pd.concat([df_TS_utc, df_geozip], axis=1)
			
			# compute local timestamp and state
			df['tz'] = zc.zip_to_tz_2( df.geo_zip )
			df_TS_local = zc.shift_tz_wrap( df, style='careful' )
			df['state'] = zc.zip_to_state_2( df.geo_zip )
			
			# compute local hour, day and day of the week
			df_locals = pd.DataFrame({'hour': zc.local_hour(df_TS_local), 'day':zc.local_day(df_TS_local), 'day_of_week':zc.local_weekday(df_TS_local)} )
			df_locals = df_locals.astype('int8')
			
			# compute the state from the zip code
			df['state'] = zc.zip_to_state_2(df.geo_zip)
			df['state'] = df.state.astype('category')
			
			# drop the bid_timestamp_utc and geo_zip columns
			df_TS_local = df_TS_local.drop(['bid_timestamp_utc', 'geo_zip'], axis=1)
			df_tz = pd.DataFrame( df['tz'] ) # save the tz column as a separate df
			df_state = pd.DataFrame( df['state'] ) # save the state column as a separate df
			df_TS_local = df_TS_local.drop(['tz', 'state'], axis=1) #drop tz and state
			
			#save things to disk (temporarily) to save RAM
			fname_TS_local = os.path.join(data_dir, 'TS_local_' + os.path.basename(f).split('.')[0] + '.gzip')
			fname_locals = os.path.join(data_dir, 'locals_' + os.path.basename(f).split('.')[0] + '.gzip')
			fname_tz = os.path.join(data_dir, 'tz_' + os.path.basename(f).split('.')[0] + '.gzip')
			fname_state = os.path.join(data_dir, 'state_' + os.path.basename(f).split('.')[0] + '.gzip')
			
			# remember the file names we use for later
			flist_TS_local.append(fname_TS_local)
			flist_locals.append(fname_locals)
			flist_tz.append(fname_tz)
			flist_state.append(fname_state)
			
			# save to disk using parquet method
			lf.temp_save(df_TS_local, os.path.join(data_dir, fname_TS_local) )
			lf.temp_save(df_locals, os.path.join(data_dir, fname_locals) )
			lf.temp_save(df_tz, os.path.join(data_dir, fname_tz) )
			lf.temp_save(df_state, os.path.join(data_dir, fname_state) )
			print(' done')
		
		# go through the saved files and combine them into a single large file
		print('saving summed gzip files ... ', end='')
		
		# save bid_timestamp_local
		df_from_each_file = (lf.temp_load(fname=f) for f in flist_TS_local)
		df_TS_local = pd.concat(df_from_each_file, ignore_index=True)
		lf.temp_save(df_TS_local, fname=os.path.join(data_dir,'bid_timestamp_local.gzip') )
		print('bid_timestamp_local.gzip ... ', end='')
		
		#save local_ordinals (hour, day, day_of_week)
		df_from_each_file2 = (lf.temp_load(fname=f) for f in flist_locals)
		df_locals = pd.concat(df_from_each_file2, ignore_index=True)
		df_locals = df_locals.astype('int8')
		lf.temp_save(df_locals, fname=os.path.join(data_dir,'local_ordinals.gzip') )
		print('local_ordinals.gzip ... ', end='')

		#save time zones
		df_from_each_file3 = (lf.temp_load(fname=f) for f in flist_tz)
		df_tz = pd.concat(df_from_each_file3, ignore_index=True)
		df_tz = df_tz.astype('category')
		lf.temp_save(df_tz, fname=os.path.join(data_dir,'tz.gzip') )
		print('tz.gzip')
		
		#save state
		df_from_each_file4 = (lf.temp_load(fname=f) for f in flist_state)
		df_state = pd.concat(df_from_each_file4, ignore_index=True)
		df_state = df_state.astype('category')
		lf.temp_save(df_state, fname=os.path.join(data_dir,'state.gzip') )
		print('state.gzip')
		
		# remove daily gzips from disk when done
		for f in flist_TS_local:
			os.remove(f)
			
		for f in flist_locals:
			os.remove(f)
			
		for f in flist_tz:
			os.remove(f)
		
		for f in flist_state:
			os.remove(f)
		
		print('temp gzip files deleted')
		print('all done!')
	return

#%% example usage
#define data directory
#data_dir = r'C:\Users\Tebe\Documents\Root Ad Data\csvs'


#reshape data from (days) to (columns) using *all columns* and save output as .gzip
#reshape_files(data_dir=data_dir, f_ext='.csv')

#reshape data from (days) to (columns) using *only specified columns* 
#mysub_cols = ['segments'] # specify which columns you want here
#reshape_files(data_dir=data_dir, f_ext='.csv', all_cols=False, sub_cols=mysub_cols)

# create local timestamp data
#local_hour_creator(data_dir=data_dir)
