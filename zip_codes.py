# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:29:35 2019

@author: Greg Smith
"""

import pandas as pd
import os
import numpy as np

class ZC():
	''' loads zip code information '''
	def __init__(self, fdir=r'C:\PythonBC\root', fname='zip_code_database.csv'):
		self.fdir = fdir
		self.fname = fname
		self.ZC = self.init_zipcodes()
		
	def init_zipcodes(self):
		''' creates a pandas dataframe of zip codes, states, and time zones
		data source: https://www.unitedstateszipcodes.org/zip-code-database/
		'''
		fdir = self.fdir
		fname = self.fname
		mycols = ['zipcode',
				  'type',
				  'decommissioned',
				  'primary_city',
				  'acceptable_cities',
				  'unacceptable_cities',
				  'state',
				  'county',
				  'timezone',
				  'area_codes',
				  'world_region',
				  'country',
				  'latitude',
				  'longitude',
				  'irs_estimated_population_2015']
		mydtypes = {'zipcode': str,
				  'type': 'category',
				  'decommissioned': np.bool,
				  'primary_city': str,
				  'acceptable_cities': str,
				  'unacceptable_cities': str,
				  'state': str,
				  'county': str,
				  'timezone': str,
				  'area_codes': 'category',
				  'world_region': str,
				  'country': str,
				  'latitude': float,
				  'longitude': float,
				  'irs_estimated_population_2015': float}
		ZC = pd.read_csv( os.path.join(fdir, fname), skiprows=1, names=mycols, dtype=mydtypes )
		ZC = ZC.set_index('zipcode')
		return ZC[['state', 'timezone', 'county', 'latitude', 'longitude']]
		
	def zip_to_tz(self, myzip):
		''' given a zip code, returns a time zone '''
		ZC = self.ZC
		try:
			ans = ZC.timezone[ZC.index == myzip].values[0]
		except:
			ans = None
		if type(ans) == float:
			ans = None # empty values in database return nan (float)
		return ans
	
	def zip_to_state(self, myzip):
		''' given a zip code, returns the US State '''
		ZC = self.ZC
		try:
			ans = ZC.state[ZC.index == myzip].values[0]
		except:
			ans = None
		if type(ans) == float:
			ans = None # empty values in database return nan (float)
		return ans

	def zip_to_county(self, myzip):
		''' given a zip code, returns the county '''
		ZC = self.ZC
		try:
			ans = ZC.county[ZC.index == myzip].values[0].split(' County')[0]
		except:
			ans = None
		if type(ans) == float:
			ans = None # empty values in database return nan (float)
		return ans
	
	def zip_to_lat_2(self, myzip):
		'''given a zip code, returns the latitude'''
		ZC = self.ZC
		return ZC.reindex(myzip).latitude.values
	
	def zip_to_lon_2(self, myzip):
		'''given a zip code, returns the longitude'''
		ZC = self.ZC
		return ZC.reindex(myzip).longitude.values
	
	def zip_to_tz_2(self, myzip):
		'''given a zip code, returns the time zone
		this function is a vectorized version of zip_to_tz, about 600x faster'''
		ZC = self.ZC
		return ZC.reindex(myzip).timezone.values

	def zip_to_state_2(self, myzip):
		'''given a zip code, returns the state
		this function is a vectorized version of zip_to_state, about 600x faster'''
		ZC = self.ZC
		return ZC.reindex(myzip).state.values
	
	def zip_to_county_2(self, myzip):
		'''given a zip code, returns the county
		this function is a vectorized version of zip_to_county, about 600x faster'''
		ZC = self.ZC
		return ZC.reindex(myzip).county.values
	
	def shift_tz_3(self, pd_df):
		'''
		shifts from UTC to local timezone
		faster than .apply() but still kinda slow (0.5s for 50k rows)
		'''
		return pd_df['bid_timestamp_utc'].groupby(pd_df.tz).apply(lambda g: g.dt.tz_convert(g.name))
	
	def strip_tz(self, pd_s):
		''' strips the timezone information from a panda series.
		SLOW! 1 second for 50k rows
		'''
		return pd_s.tz_localize(None)
	
	def strip_tz_2(self, pd_s):
		'''
		- shifts from UTC to local timezone
		- same functionality as strip_tz, but this one handles errors better
		- if a non-TimeStamp object is passed, it returns a 'fake' date instead
		of crashing
		'''
		if type(pd_s) == float: # there was an error computing the timezone
			# there was an error computing the timezone
			# return a nonsense time and continue
			return pd.Timestamp(2000, 1, 1, 12)
		else: # everything is fine
			return pd_s.tz_localize(None)
	
	def shift_tz_wrap(self, pd_df, style='normal'):
		'''wrapper function for strip_tz and shift_tz_3'''
		pd_df['bid_timestamp_local'] = self.shift_tz_3(pd_df)
		if style == 'normal':
			pd_df['bid_timestamp_local'] = pd_df['bid_timestamp_local'].apply( self.strip_tz )
		if style == 'careful':
			pd_df['bid_timestamp_local'] = pd_df['bid_timestamp_local'].apply( self.strip_tz_2 )			
		return pd_df
	
	def local_hour(self, pd_df):
		'''computes the hour of the local time'''
		return pd_df["bid_timestamp_local"].dt.hour.values.astype('int8')
	
	def local_day(self, pd_df):
		'''computes the day (i.e., 11 for May 11) of the local time'''
		return pd_df["bid_timestamp_local"].dt.day.values.astype('int8')

	def local_weekday(self, pd_df):
		'''computes the weekday (i.e., Tuesday) of the local time
		represented as an integer.
		Monday=0, ... Sunday=6
		'''
		return pd_df["bid_timestamp_local"].dt.weekday.values.astype('int8')

#%% useage examples

### single-line use
#zc = ZC(fdir='') #initialize
#zc.zip_to_tz('43212') # 'America/New_York'
#zc.zip_to_state('43212') # 'OH'
#zc.zip_to_county('43212') # 'Frankin'

# vectorized versions require a list or pd.series
#zc.zip_to_tz_2(['43212']) # 'America/New_York'
#zc.zip_to_state_2(['43212']) # 'OH'
#zc.zip_to_county_2(['43212']) # 'Frankin'


### using with pandas:
# load data
#df = load_data(fname=fname, data_dir=data_dir, nrows=50000)

# add tz and state columns
#df['tz'] = zc.zip_to_tz_2(df.geo_zip)
#df['state'] = zc.zip_to_state_2(df.geo_zip)
#df['tz'] = df.tz.astype('category')
#df['state'] = df.state.astype('category')

# compute local time
#df = zc.shift_tz_wrap(df)
#df['hour'] = zc.local_hour(df)
