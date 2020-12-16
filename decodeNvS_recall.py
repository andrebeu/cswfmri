#!/usr/bin/env python
# coding: utf-8

# # Decoding state information

# - extract information from files in `recall_transcriptions` folder.
#     - see `behav` notebook

# In[ ]:


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
from sklearn.linear_model import LogisticRegression
pd.options.display.max_rows = 200


# In[ ]:


save_dir = 'data/analyses/xval_NvSrecall/'


# # load dataframes with timing and order information

# In[ ]:


view_df = pd.read_csv('deriv/view_df.csv',index_col=0)
recall_df = pd.read_csv('deriv/recall_df.csv',index_col=0)
view_df.iloc[:200]


# # roi

# In[ ]:


def load_sub_roi(sub_num,roi_name,task):
  fpath = "sub-%i_task-%s_roi-%s.npy" %(sub_num,task,roi_name)
  return np.load('data/fmri/masked/'+fpath)

ROI_NAME_L = [
  'SnPM_filtered_FDR',
  'rglasser_AT_net',
  'rglasser_MP_net',
  'rglasser_MTN_net',
  'rglasser_PM_net',
  'rhippocampusAAL', 
]


# In[ ]:


def load_sub_roi(sub_num,task,roi_name,filt,motion):
  if filt==0: filt=None
  fpath = "sub-%i_task-%s_roi-%s-filter_%s-motioncorr_%s.npy"%(sub_num,task,roi_name,filt,bool(motion))
  return np.load('data/fmri/roi_act/'+fpath)


# In[ ]:


def get_train_info(sub_num):
  """ returns two lists with 
  training data for sub `sub_num`
  TR_L is a list of TRs from viewing
  """
  TR_L = []
  ytarget_L = []
  sub_view_df = view_df[view_df.sub_num == sub_num]
  for wed_num in np.arange(12):
    sub_wed_view_df = sub_view_df[sub_view_df.wed_num == wed_num].sort_values('onsetTR')
    schema = sub_wed_view_df.wed_schema.unique()[0]
    onset_TR = sub_wed_view_df
    onTR,offTR = sub_wed_view_df.iloc[0].onsetTR,sub_wed_view_df.iloc[-1].offsetTR
    wed_TRs = np.arange(onTR,offTR)
    TR_L.extend(wed_TRs)
    ytarget_L.extend(np.repeat(schema=='N',len(wed_TRs)))
  assert len(ytarget_L)==len(TR_L)
  return TR_L,ytarget_L

def get_test_info(sub_num):
  """
  """  
  sub_recall_df = recall_df[recall_df.sub_num == sub_num]
  sub_view_df = view_df[view_df.sub_num == sub_num]
  TR_L = []
  ytarget_L = []
  for wed_id in sub_recall_df.wed_id.unique():
    # select subset of df for given wedding
    sub_wed_view_df = sub_view_df[sub_view_df.wed_id == wed_id]
    sub_wed_recall_df = sub_recall_df[sub_recall_df.wed_id == wed_id]
    # wedding TR
    recall_on_TR = sub_wed_recall_df.onsetTR.unique()[0]
    recall_off_TR = sub_wed_recall_df.offsetTR.unique()[0]
    wed_recall_TRs = np.arange(recall_on_TR,recall_off_TR)
    # wedding schema
    schema = sub_wed_view_df.wed_schema.unique()[0]
    # extend L
    ytarget_L.extend(np.repeat(schema =='N',len(wed_recall_TRs)))
    TR_L.extend(wed_recall_TRs)
  assert len(ytarget_L)==len(TR_L)
  return TR_L,ytarget_L


# In[ ]:


sub_nums = np.arange(45)
roi_names = ROI_NAME_L
filt_L = [0,128,480]
motion_L = [False,True]

for sub_num,roi_name,filt,motion in itertools.product(sub_nums,roi_names,filt_L,motion_L):
  try:
    act_view = load_sub_roi(sub_num,'videos',roi_name,filt,motion)
    act_recall = load_sub_roi(sub_num,'recall2',roi_name,filt,motion)
  except:
    print('err loading',sub_num,roi_name,filt,motion)
    continue
  try:
    TR_L_train,ytarget_train = get_train_info(sub_num)
    TR_L_test,ytarget_test = get_test_info(sub_num)
  except:
    print('err finding TRs',sub_num,roi_name,filt,motion)
    continue
    
  print('loaded',sub_num,roi_name,filt,motion)
  # extract roi TRs
  xact_train = act_view[TR_L_train]
  xact_test = act_recall[TR_L_test]
  # scale data
  scaler = StandardScaler()
  xact_train = scaler.fit_transform(xact_train)
  xact_test = scaler.transform(xact_test)
  # fit classifier
  clf = sklearn.linear_model.LogisticRegression(solver='liblinear',C=1.00)
  clf.fit(xact_train,ytarget_train)
  # eval calssifier
  yhat_test = clf.predict_proba(xact_test)[:,1]
  # save
  save_fpath = "predict_proba-sub_%i-roi_%s-filter_%s-motion_%s"%(sub_num,roi_name,filt,motion)
  np.save(save_dir+save_fpath,yhat_test)

