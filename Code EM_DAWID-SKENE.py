# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:48:58 2022

@author: Nicolas
"""

#%%import numpy as np
import pandas as pd

# Load the Data from the google drive link via a shareable link
data1 = pd.read_csv('https://drive.google.com/uc?id=1F3eLDpLDKVCmpV2Tvvn7H5U9glPyeMWp', header=0, dtype='category')
# We know all the data is categorical, so let us change it as such:
data = data1.astype('category')

#%%
from sklearn.metrics import confusion_matrix
 
confusions = {} # Store in a dict
for col in range(5):
  sub_data = data.loc[:, ['GT', 'A{}'.format(col)]] # Get the two columns of interest
  sub_data.dropna(axis='index', how='any', inplace=True)   # Drop rows which this annotator did not label
  confusions[col] = confusion_matrix(sub_data['GT'], sub_data['A{}'.format(col)], normalize='true')

#%%
#from mpctools.extensions import mplextpip pb du package lapsolver 
#(pb de setup.py : Building wheel for lapsolver (setup.py) ... error)

