# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:21:42 2019

@author: Bryan
"""

'''*************************************************************
|For those of you unfamiliar with loading other python files as| 
|python molules simply place the file containing the code you  |
|wish to use as a python module in the same directory as your  |
|main python code, then import using import file name. As you  |
|can see you do not need to include the file extension. I have |
|included an example of this using Greg's load_file.py code.   |
|This allows you to group useful sets of functions together in |
|a module ultimately making your code more readable.           |
*************************************************************'''

import load_file as lf

fname = '2019-04-27.csv'
fname_zip = '2019-04-27.zip'
data_dir = r'D:\Root.Ad.Auction\Data\1'

df = lf.load_data(fname=fname, data_dir=data_dir, Verbose=True)

    
    