#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

import sklearn
import brainiak
import nilearn as nl
from nilearn import image, plotting, input_data

from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler


# In[2]:


from sklearn.linear_model import LogisticRegression
pd.options.display.max_rows = 200


# # load dataframes with timing and order information

# In[3]:


wed_df = pd.read_csv('deriv/wed_df.csv',index_col=0)
wed_df.iloc[55:65]


# In[4]:


def load_sub_roi(sub_num,roi_name,task):
  fpath = "sub-%i_task-%s_roi-%s.npy" %(sub_num,task,roi_name)
  return np.load('data/fmri/masked/'+fpath)


# In[5]:


def get_data(sub_num,roi_name,task):
  """
  task: [videos,recall2]
  returns the func data for given sub/roi/task for all 12 weddings
  """
  try:
    sub_roi_act = load_sub_roi(sub_num,roi_name,task)
    sub_wed_df = wed_df[wed_df.sub_num==sub_num]
  except:
    print('err loading sub',sub_num)
    return None
  Xact_L,ytarget_L = [],[]
  stimstr_L = []
  for idx,df_row in sub_wed_df.iterrows():
    Xact_wed = sub_roi_act[df_row.loc['onset_%s'%task]:df_row.loc['offset_%s'%task]]
    ytarget_wed = np.repeat(int(df_row.wed_class == 'N'),len(Xact_wed))
    Xact_L.append(Xact_wed)
    ytarget_L.append(ytarget_wed)
    # string identifying test sequences
    stimstr = "wed_%i-class_%s"%(df_row.wed_id,df_row.wed_class)
    stimstr_L.append(stimstr)
  return Xact_L,ytarget_L,stimstr_L


# In[6]:


roi_name= 'rglasser_PM_net'
clf_c = 1.00

for sub_num in range(30,39):
  print('sub',sub_num)
  ## train data
  try:
    Xact_train_L,ytarget_train_L,stimstr_L = get_data(sub_num,roi_name,'videos')
  except:
    continue
  ytarget_train = np.concatenate(ytarget_train_L)
  Xact_train = np.concatenate(Xact_train_L)
  ## test data
  Xact_test_L,ytarget_test_L,stimstr_L = get_data(sub_num,roi_name,'recall2')
  ## normalize
  scaler = StandardScaler()
  Xact_train = scaler.fit_transform(Xact_train)
  Xact_test_L = [scaler.transform(Xact) for Xact in Xact_test_L]
  ## fit classifier
  clf = sklearn.linear_model.LogisticRegression(solver='liblinear',C=clf_c)
  clf.fit(Xact_train,ytarget_train)
  ## EVAL LOOP: loop over 12 weddings for eval
  for idx_test in range(12):
    stimstr = stimstr_L[idx_test]
    # eval data for given wedding
    Xact_test_wed = Xact_test_L[idx_test]
    ytarget_test_wed = np.unique(ytarget_test_L[idx_test])
    # fit classifier
    yhat_wed = clf.predict_proba(Xact_test_wed)[:,ytarget_test_wed]
    np.save('deriv/analyses/NvS_train_view_test_recall/sub_%i-%s'%(sub_num,stimstr),yhat_wed)

