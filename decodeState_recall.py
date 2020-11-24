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


# ### recall legend
# - 3,4: layer 2
# - 5,6: layer 3
# - 7,8: layer 4

# In[2]:


view_df = pd.read_csv('deriv/view_df.csv',index_col=0)
# view_df


# In[3]:


wed_df = pd.read_csv('deriv/wed_df.csv',index_col=0)
wed_df.path = wed_df.path.fillna('NA')
# wed_df.iloc[:200]


# In[4]:


recall_df = pd.read_csv('deriv/recall_df.csv',index_col=0)
# recall_df


# # load dataframes with timing and order information

# In[5]:


def load_sub_roi(sub_num,roi_name,task):
  fpath = "sub-%i_task-%s_roi-%s.npy" %(sub_num,task,roi_name)
  return np.load('data/fmri/masked/'+fpath)


# # view-recall state classifier
# - different decoder for each sub/layer
# -- each sub has 3 decoders

# ### classifier training 

# In[6]:


""" 
build training dataset
"""

pathlayer2label_D = {
  ('NA',2):'a',
  ('NA',3):'a',
  ('NA',4):'a',
  ('NB',2):'b',
  ('NB',3):'b',
  ('NB',4):'b',
  ('SA',2):'a',
  ('SA',3):'b',
  ('SA',4):'a',
  ('SB',2):'b',
  ('SB',3):'a',
  ('SB',4):'b',
}

def get_training_info(sub_num,layer_num):
  """
  for given subject/layer 
  returns info needed to train state classifier
    the TRs when viewing layer (for all 12 weddings)
    and the labels of the states for that layer
  """
  ytarget_L = []
  TR_L = []
  # wed_df contains labels for given wedding 
  # select subject specific rows of wed_df
  sub_wed_df = wed_df[wed_df.sub_num==sub_num]
  for wed_num in range(12):
    path = sub_wed_df[sub_wed_df.wed_view_num==wed_num].path.values[0]
    wed_bool = (view_df.wed_num == wed_num) 
    layer_bool = (view_df.vid_str.str[:len('vid1')] == 'vid%i'%layer_num)
    # TRs for given state (for given sub/layer)
    onsetTR,offsetTR = view_df[wed_bool&layer_bool].loc[:,('onset_TR','offset_TR')].values[0]
    state_TRs = np.arange(onsetTR,offsetTR)
    # state label
    state_label = pathlayer2label_D[(path,layer_num)]
    # 
    TR_L.extend(state_TRs)
    ytarget_L.extend(np.repeat(state_label=='a',len(state_TRs)))
  return np.array(TR_L),np.array(ytarget_L)


# In[7]:


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


# In[11]:


""" 
train and test classifier
"""

roi_name= 'rglasser_PM_net'
clf_c = 1.00

L = []
for sub_num,layer_num in itertools.product(np.arange(45),range(2,5)):
  print('s',sub_num,'l',layer_num)
  # load fmri data; 
  try: # check that fmri files exist
    sub_roi_view = load_sub_roi(sub_num,roi_name,'videos')
    sub_roi_recall = load_sub_roi(sub_num,roi_name,'recall2')
    assert len(sub_roi_view)
    assert len(sub_roi_recall)
  except:
    print('err loading sub',sub_num)
    continue
  ## build train/test datasets
  # train
  train_TRs,Ytrain = get_training_info(sub_num,layer_num)
  Xtrain = sub_roi_view[train_TRs,:]
  # test  
  test_TRs,Ytest = get_test_info(sub_num,layer_num)
  Xtest = sub_roi_recall[test_TRs,:]
  # check if recall data exists
  if not len(Xtest): 
    print('no recall data. sub',sub_num,'layer',layer_num)
    continue
  ## normalize
  scaler = StandardScaler()
  Xtrain = scaler.fit_transform(Xtrain)
  Xtest = scaler.transform(Xtest)
  ## fit classifier
  clf = sklearn.linear_model.LogisticRegression(solver='liblinear',C=clf_c)
  clf.fit(Xtrain,Ytrain)
  # eval classifier
  yhat = clf.predict_proba(Xtrain)
  score = clf.score(Xtest,Ytest)
  # record
  D = {}
  D['num_test_samples']=len(Ytest)
  D['sub_num']=sub_num
  D['layer']=layer_num
  D['score']=score
  L.append(D)


# In[9]:


results = pd.DataFrame(L)
results.to_csv('data/analyses/decodeState_trainView_testRecall.csv')

