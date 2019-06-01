# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:41:41 2019

@author: Greg Smith
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def make_countplot(pd_df, col='platform_device_make', count='clicks', order=True):
	'''make a histogram
	pd_df: pandas dataframe that has the data
	col: the thing we are counting
	count: optional. a 2nd axis. for example, count=clicks (True or False) or None (disable)
	order: if True, x-axis will be ordered by counts
	'''
	fig, ax = plt.subplots()
	if order:
		if count == None:
			sns.countplot(x=col,
			  data=pd_df,
			  order=pd_df[col].value_counts().index,
			  ax=ax)	
		else:
			sns.countplot(x=col,
			  data=pd_df,
			  order=pd_df[col].value_counts().index,
			  hue=count,
			  ax=ax)
	else:
		if count == None:
			sns.countplot(x=col,
				  data=pd_df,
				  hue=count,
				  ax=ax)			
		else:	
			sns.countplot(x=col,
				  data=pd_df,
				  hue=count,
				  ax=ax)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
	ax.set_yscale('log')
	plt.tight_layout()
	return ax

def ratio_plot(pd_df, col='day', **kwargs):
	df_clicks = pd_df.groupby(col).sum()['clicks'].to_frame()
	df_clicks.reset_index(level=0, inplace=True)

	# number of bids per col
	df_bids = pd_df.groupby(col).count()['clicks'].to_frame()
	df_bids.reset_index(level=0, inplace=True)
	df_bids.rename(index=str, columns={col: col, 'clicks': "bids"})
	
	# number of installs per col
	df_installs = pd_df.groupby(col).sum()['installs'].to_frame()
	df_installs.reset_index(level=0, inplace=True)
	df_installs.rename(index=str, columns={col: col, "clicks": "installs"})

	# build new dataframe
	bids = df_bids.clicks.values
	clicks = df_clicks.clicks.values
	installs = np.asarray(df_installs.installs)
	state = df_clicks[col].values
	clickrate = 100*np.divide(clicks, bids)
	installrate = 100*np.divide(installs, bids)
	frames = {col: state, "bids": bids, "clicks": clicks, "installs": installs, "clickrate":clickrate, "installrate": installrate}
	df_rate = pd.DataFrame(data=frames)


	fig, ax = plt.subplots(2,2, sharex=True)
	
	df_rate.plot(x=col, y='clicks', ax=ax[0,0], **kwargs)
	df_rate.plot(x=col, y='installs', ax=ax[0,1], **kwargs)
	df_rate.plot(x=col, y='clickrate', ax=ax[1,0], **kwargs)
	df_rate.plot(x=col, y='installrate', ax=ax[1,1], **kwargs)
	plt.tight_layout()
	plt.show()
	return ax
	
def utc_vs_local(pd_df):
	fig, ax = plt.subplots(2,1, sharex=True)
	ax[0].hist(pd_df.hour, bins=list(np.arange(0,25)))
	ax[0].set_ylabel('local hour')
	ax[1].hist(pd_df.bid_timestamp_utc.dt.hour, bins=list(np.arange(0,25)))
	ax[1].set_ylabel('utc hour')
	ax[0].set_xlim([0,24])
	ax[0].set_title('correcting for local time')
	return ax
