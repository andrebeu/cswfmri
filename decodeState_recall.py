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

from sklearn.linear_model import LogisticRegression
pd.options.display.max_rows = 200


# # load dataframes with timing and order information

# In[2]:


view_df = pd.read_csv('deriv/view_df.csv',index_col=0)
view_df.iloc[:200]


# In[3]:


recall_df = pd.read_csv('deriv/recall_df.csv',index_col=0)
# recall_df


# # form training and testing datasets

# ### classifier training 

# In[4]:


sub_num,layer_num = 33,3
def get_training_info(sub_num,layer_num):
  # find df rows corresponding to sub/layer
  layer_bool = view_df.state.str[0]==str(layer_num)
  sub_layer_view_df = view_df[(view_df.sub_num == sub_num) & layer_bool]
  # extract TRs and labels 
  TR_L = []
  ytarget_L = []
  for idx,row in sub_layer_view_df.iterrows():
    TRs = np.arange(row.onset_TR,row.offset_TR)
    TR_L.extend(TRs)
    ytarget_L.extend(np.repeat(row.state[1]=='a',len(TRs)))
  return np.array(TR_L),np.array(ytarget_L)


# In[5]:


""" 
build testing dataset
""" 

## dict mapping (layer,state):recall_transcript_code
recall_label_D = {
  (2,'a'):3,
  (2,'b'):4,
  (3,'a'):5,
  (3,'b'):6,
  (4,'a'):7,
  (4,'b'):8
}

def get_test_info(sub_num,layer_num):
  """ 
  build testing dataset
  find TRs during recall when sub is recalling given state+layer
  along with labels for these recall TRs when recalling layer
  """
  ytarget = []
  XTRs = []
  sub_recall_df = recall_df[recall_df.sub_num==sub_num]
  for state_id in ['a','b']:
    # from layer+state get transcript_code
    recall_code = recall_label_D[(layer_num,state_id)]
    # find TRs where sub talks about layer+state
    TRs_state = sub_recall_df[sub_recall_df.recall==recall_code].index.values
    XTRs.extend(TRs_state)
    ytarget.extend(np.repeat(state_id,len(TRs_state)))
  return XTRs,np.array(ytarget)=='a'


# # train-test loop

# In[6]:


def load_sub_roi(sub_num,roi_name,task):
  fpath = "sub-%i_task-%s_roi-%s.npy" %(sub_num,task,roi_name)
  return np.load('data/fmri/masked/'+fpath)


# In[7]:


ROI_NAME_L = [
  'SnPM_filtered_FDR',
  'rglasser_AT_net',
  'rglasser_MP_net',
  'rglasser_MTN_net',
  'rglasser_PM_net',
  'rhippocampusAAL', 
]


# In[8]:


roi_name= 'rglasser_PM_net'


# In[9]:


""" 
train and test classifier
"""

L = []
for roi_name,sub_num,layer_num in itertools.product(ROI_NAME_L,np.arange(45),range(2,5)):
  print('r',roi_name,'sub',sub_num,'layer',layer_num)
  # load fmri data; 
  try: # check that fmri files exist
    sub_roi_view = load_sub_roi(sub_num,roi_name,'videos')
    sub_roi_recall = load_sub_roi(sub_num,roi_name,'recall2')
    assert len(sub_roi_view)
    assert len(sub_roi_recall)
  except:
    print('err loading roi data')
    continue
  ## build train/test datasets
  try:
    train_TRs,Ytrain = get_training_info(sub_num,layer_num)
    Xtrain = sub_roi_view[train_TRs,:] 
    test_TRs,Ytest = get_test_info(sub_num,layer_num)
    Xtest = sub_roi_recall[test_TRs,:]
  except:
    print('err finding info')
    continue
  # check if recall data exists
  if not len(Xtest): 
    print('no recall data. sub',sub_num,'layer',layer_num)
    continue
  ## normalize
  scaler = StandardScaler()
  Xtrain = scaler.fit_transform(Xtrain)
  Xtest = scaler.transform(Xtest)
  ## fit classifier
  clf = sklearn.linear_model.LogisticRegression(solver='liblinear',C=1.00)
  clf.fit(Xtrain,Ytrain)
  # eval classifier
  yhat = clf.predict_proba(Xtrain)
  score = clf.score(Xtest,Ytest)
  # record data
  D = {}
  D['num_test_samples']=len(Ytest)
  D['sub_num']=sub_num
  D['layer']=layer_num
  D['score']=score
  L.append(D)

## 
results = pd.DataFrame(L)


# In[10]:


Nsubs = len(results.sub_num.unique())
results.to_csv('data/analyses/decodeState_trainView_testRecall-N%i.csv'%Nsubs)

