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

#%%
import statistics
set = [1,2,3,4,5,6]
statistics.mode(set)
import numpy as np

def majority_vote(annotations):
    """
    Majority Vote Method
 
    :param annotations: The annotations (Samples by Labels): dataframe
    :return: Majority-Vote for each row
    """
    mode = annotations.mode(axis='columns', dropna=True).astype(float)
    #return a data frame with choices for each row
    return np.where(mode.count(axis='columns') == 1, mode[0], np.NaN)
    #return index of respons under condition one choice is possible, set Nan else
data['MV'] = majority_vote(data[['A0', 'A1', 'A2', 'A3', 'A4']])
data['MV'].count()

from sklearn.metrics import accuracy_score
# Remove NaNs.
sub_data_mv = data[['GT', 'MV']].dropna(axis='index', how='any')
accuracy_mv = accuracy_score(sub_data_mv['GT'].astype(int), sub_data_mv['MV'].astype(int))

