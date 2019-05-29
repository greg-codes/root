# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:17:01 2019

@author: Greg Smith

"""

import os, glob, sys
sys.path.append(os.getcwd()) # add cwd to path
import load_file as lf
import pandas as pd
#import numpy as np
#from zipfile import ZipFile

def reshape_files(data_dir=r'C:\PythonBC\RootData', f_ext='.csv', big_zip=False, **kwargs):
    '''
    'reshapes' the data files. the original format of the data files provided
    by root gives us one day's worth of data per file, with all variables
    present. this function rearranges the format of the files so that each file
    contains data from all days, but only one column. files are saved via the
    parquet method (non-human-readable) as <column>.gzip for each column.
    
    running this file will this 'reshaping' on all files in data_dir 
    matching *.f_ext.
    
    input parameters:
        data_dir: location of files
        f_ext: filename extension ('.csv' or '.zip') of files to be loaded. 
            Ignored if big_zip == True.
        big_zip: True or False. If False, will perform the reshaping on all 
            files in data_dir matching *f_ext. If True, the function will
            perform the reshaping on bigzipname (hard-coded).
        
    '''
    # define the file name of the large zip file. assumed structure is a .zip
    # file with many .csv files inside
    bigzipname = 'root_ad_auction_dataset_all_data.zip'

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
    unwanted = ['auction_id', 'platform_os', 'day', 'month', 'year', 'hour']
    
    #drop the unwanted columns from mycols
    mycols = [ele for ele in mycols if ele not in unwanted]

    # get a list of all the csv files in the data director
    myfiles = glob.glob( os.path.join(data_dir, '*'+f_ext) )

    # sort <myfiles>
    myfiles = sorted( myfiles )
    
    # remove the big zip file from <myfiles>
    if bigzipname in myfiles:
        myfiles.remove(bigzipname)

    if big_zip: # load the files from the mega zip file
        print(f"this code not written yet. let's pretend this opened {bigzipname}")
    else: # load files from individual csv/zip files
        for col in mycols[1:]: # loop through all items in mycols
            print(f'loading {col}...', end='')
            df_from_each_file = (lf.load_data(data_dir=data_dir, fname=f, all_cols=False, sub_cols=[col], **kwargs) for f in myfiles)
            df = pd.concat(df_from_each_file, ignore_index=True)
            #df = load_data(all_cols=False, sub_cols=[col], nrows=5000)
            print('done')
            myfname = col + '.gzip'
            lf.temp_save(df, os.path.join(data_dir, myfname) ) # save to disk using parquet method
            print(f'   {myfname} saved')
    print('all done!')
    return

#%% useage example

#reshaping the files (only do once)
#data_dir=r'C:\PythonBC\RootData'
#reshape_files()

##load dataframes from disk when you want to use them
#df_installs = lf.temp_load( os.path.join(data_dir, 'installs.gzip') ) # 37.8 ms to 4 days worth of data
#df_geo_zip = lf.temp_load( os.path.join(data_dir, 'geo_zip.gzip') )