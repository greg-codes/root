# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:33:07 2019

@author: Greg Smith
"""
#%% comparing the speed of different time methods
# source: https://github.com/sanand0/benchmarks/blob/master/date-parse/date-parse.py

import datetime
import pandas as pd
import timeit

mytime = '2019-04-27 20:28:00.013'
year = mytime[0:4]
month = mytime[5:7]
day = mytime[8:10]
hour = mytime[11:13]
minute = mytime[14:16]
sec = mytime[17:19]
ms = mytime[20:24]
pd_mytime = pd.Series([mytime]*1000)


#################################
## operations on strings
#################################

%timeit datetime.datetime( int(year), int(month), int(day), int(hour), int(minute), int(sec), 1000*int(ms) )
#1.63 µs ± 70.4 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

%timeit datetime.datetime.strptime(mytime, '%Y-%m-%d %H:%M:%S.%f')
#12.3 µs ± 416 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

%timeit pd.to_datetime(mytime)
#112 µs ± 2 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit pd.to_datetime(mytime,utc=True)
#128 µs ± 1.93 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit pd.to_datetime(mytime, utc=True, infer_datetime_format=True)
#567 µs ± 12.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit pd.to_datetime(mytime, format='%Y-%m-%d %H:%M:%S.%f',utc=True)
#131 µs ± 1.89 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

#################################
## operations on panda series
#################################

%timeit pd.to_datetime(pd_mytime, utc=True)
#20.5 ms ± 1.23 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit pd.to_datetime(pd_mytime, utc=True, format='%Y-%m-%d %H:%M:%S.%f' )
#21.1 ms ± 2.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit pd.to_datetime(pd_mytime, utc=True, infer_datetime_format=True )
#25.5 ms ± 1.84 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

def my_todatetime(s):
    #doesn't work
    return pd.to_datetime(s, utc=True, format='%Y-%m-%d %H:%M:%S.%f' )

%timeit pd_mytime.apply( my_todatetime )
# did not finish in 30 seconds...

%timeit pd.Series([datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in pd_mytime.values])

np.datetime64(pd_mytime)
%timeit pd.Series([datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in pd_mytime.values])
#1.23 s ± 47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit pd_mytime.apply(lambda v: datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(sec), 1000*int(ms)))
#183 ms ± 2.13 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)