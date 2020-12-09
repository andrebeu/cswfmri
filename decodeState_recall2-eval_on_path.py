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
recall_df


# # form training and testing datasets

# ### classifier training 

# In[4]:


## dict mapping (layer,state):recall_transcript_code
recall_label_D = {
  (2,'a'):3,
  (2,'b'):4,
  (3,'a'):5,
  (3,'b'):6,
  (4,'a'):7,
  (4,'b'):8
}

""" 
when constructing training data, 
I assume that 'a' states are positive
the first two are paths with 'a' states
the last two are paths with 'b' states
"""
layer2_paths_D = {
  2: ['Na','Sa','Nb','Sb'],
  3: ['Na','Sb','Nb','Sa'],
  4: ['Na','Sa','Nb','Sb']
}


# In[5]:


def get_training_info(sub_num,layer_num):
  # find df rows corresponding to sub/layer
  layer_bool = view_df.state.str[0]==str(layer_num)
  sub_bool = (view_df.sub_num == sub_num)
  sub_layer_view_df = view_df[sub_bool & layer_bool]
  # extract TRs and labels 
  TR_L = []
  ytarget_L = []
  for idx,row in sub_layer_view_df.iterrows():
    TRs = np.arange(row.onset_TR,row.offset_TR)
    TR_L.extend(TRs)
    ytarget_L.extend(np.repeat(row.state[1]=='a',len(TRs)))
  return np.array(TR_L),np.array(ytarget_L)


# In[6]:


def get_test_info(sub_num,layer_num):
  sub_bool = (recall_df.sub_num==sub_num)
  sub_recall_df = recall_df[sub_bool]
  XTRs = []
  ylabels = []
  for pos_path in layer2_paths_D[layer_num][:2]: 
    pos_path_TRs = sub_recall_df[sub_recall_df.path == pos_path].TR.values
    XTRs.extend(pos_path_TRs)
    ylabels.extend(np.repeat(True,len(pos_path_TRs)))
  for neg_path in layer2_paths_D[layer_num][2:]: 
    neg_path_TRs = sub_recall_df[sub_recall_df.path == neg_path].TR.values
    XTRs.extend(neg_path_TRs)
    ylabels.extend(np.repeat(False,len(neg_path_TRs)))
  return np.array(XTRs),np.array(ylabels)


# # main

# In[7]:


def load_sub_roi(sub_num,roi_name,task):
  fpath = "sub-%i_task-%s_roi-%s.npy" %(sub_num,task,roi_name)
  return np.load('data/fmri/masked/'+fpath)


# In[8]:


ROI_NAME_L = [
  'SnPM_filtered_FDR',
  'rglasser_AT_net',
  'rglasser_MP_net',
  'rglasser_MTN_net',
  'rglasser_PM_net',
  'rhippocampusAAL', 
]


# In[9]:


def get_classifier_data(sub_num,roi_name,layer_num):
  """ 
  given info, returns train and test data
  """
  # load fmri data; 
  try: # check that fmri files exist
    act_view = load_sub_roi(sub_num,roi_name,'videos')
    act_recall = load_sub_roi(sub_num,roi_name,'recall2')
    assert len(act_view)
    assert len(act_recall)
  except:
    print('err loading roi data')
    return None
  
  ## train data
  try: # train
    XTR_train,ylabel_train = get_training_info(sub_num,layer_num)
    Xtrain = act_view[XTR_train,:]
  except:
    print('err getting classifier train data')
    return None
  
  ## test data
  try: 
    XTR_test,ylabel_test = get_test_info(sub_num,layer_num)  
    Xtest = act_recall[XTR_test,:]
  except:
    print('err getting classifier test data')
    return None
  # check if recall data exists
  try:
    assert len(Xtest)
  except:
    print('no recall data')
    return None
  
  return Xtrain,ylabel_train,Xtest,ylabel_test


# In[10]:


""" 
train and test classifier
"""

L = []

# data level vars
for roi_name,sub_num in itertools.product(ROI_NAME_L,np.arange(45)):
  # analysis level vars
  for layer_num in np.arange(2,5):
    print('roi',roi_name,'sub',sub_num,'layer',layer_num)
    
    ## LOAD DATA
    data = get_classifier_data(sub_num,roi_name, layer_num)
    if data == None: continue
    Xtrain,Ytrain,Xtest,Ytest = data
     
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
    
    ## record data
    D = {}
    D['sub_num']=sub_num
    D['roi']=roi_name
    D['layer']=layer_num
    D['num_test_samples']=len(Ytest)
    D['score']=score  
    L.append(D)

## 
results = pd.DataFrame(L)


# In[11]:


results


# In[12]:


Nsubs = len(results.sub_num.unique())
results.to_csv('data/analyses/decodeState_trainView_testRecall-eval_on_path-N%i.csv'%Nsubs)

