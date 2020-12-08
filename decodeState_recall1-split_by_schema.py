#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# In[3]:


view_df = pd.read_csv('deriv/view_df.csv',index_col=0)
view_df.iloc[:200]


# In[4]:


recall_df = pd.read_csv('deriv/recall_df.csv',index_col=0)
recall_df


# # form training and testing datasets

# ### classifier training 

# In[5]:


## dict mapping (layer,state):recall_transcript_code
recall_label_D = {
  (2,'a'):3,
  (2,'b'):4,
  (3,'a'):5,
  (3,'b'):6,
  (4,'a'):7,
  (4,'b'):8
}


# In[6]:


def get_training_info(sub_num,layer_num,schema):
  # find df rows corresponding to sub/layer
  layer_bool = view_df.state.str[0]==str(layer_num)
  schema_bool = (view_df.schema == schema)
  sub_layer_view_df = view_df[(view_df.sub_num == sub_num) & layer_bool & schema_bool]
  # extract TRs and labels 
  TR_L = []
  ytarget_L = []
  for idx,row in sub_layer_view_df.iterrows():
    TRs = np.arange(row.onset_TR,row.offset_TR)
    TR_L.extend(TRs)
    ytarget_L.extend(np.repeat(row.state[1]=='a',len(TRs)))
  return np.array(TR_L),np.array(ytarget_L)


# In[7]:


def get_test_info(sub_num,layer_num,schema):
  """ 
  build testing dataset
  find TRs during recall when sub is recalling given state+layer
  along with labels for these recall TRs when recalling layer
  """
  ytarget = []
  XTRs = []
  sub_bool = (recall_df.sub_num==sub_num)
  schema_bool = (recall_df.schema==schema)
  sub_recall_df = recall_df[sub_bool & schema_bool]
  for state_id in ['a','b']:
    # from layer+state get transcript_code
    recall_code = recall_label_D[(layer_num,state_id)]
    # find TRs where sub talks about layer+state
    TRs_state = sub_recall_df[sub_recall_df.recall==recall_code].TR.values
    XTRs.extend(TRs_state)
    ytarget.extend(np.repeat(state_id,len(TRs_state)))
  return XTRs,np.array(ytarget)=='a'


# # STICK THIS IN MAIN LOOP

# # main

# In[21]:


def load_sub_roi(sub_num,roi_name,task):
  fpath = "sub-%i_task-%s_roi-%s.npy" %(sub_num,task,roi_name)
  return np.load('data/fmri/masked/'+fpath)


# In[22]:


ROI_NAME_L = [
  'SnPM_filtered_FDR',
  'rglasser_AT_net',
  'rglasser_MP_net',
  'rglasser_MTN_net',
  'rglasser_PM_net',
  'rhippocampusAAL', 
]


# In[26]:


def get_classifier_data(sub_num,roi_name, layer_num,schema_train,schema_test):
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
  ## build train/test datasets
  try:
    XTR_train,ylabel_train = get_training_info(sub_num,layer_num,schema_train)
    XTR_test,ylabel_test = get_test_info(sub_num,layer_num,schema_test)
    # 
    Xtrain = act_view[XTR_train,:] 
    Xtest = act_recall[XTR_test,:]
    # check if recall data exists
    assert len(Xtest)
  except:
    print('err finding info to build classifier dataset')
    return None
  return Xtrain,ylabel_train,Xtest,ylabel_test


# In[27]:


""" 
train and test classifier
"""

L = []

# data level vars
for roi_name,sub_num in itertools.product(ROI_NAME_L,np.arange(45)):
  # analysis level vars
  for schema_train,schema_test,layer_num in itertools.product(['N','S'],['N','S'],np.arange(2,5)):
    print('roi',roi_name,'sub',sub_num,'layer',layer_num,'sch_train',schema_train,'sch_test',schema_test)
    
    ## LOAD DATA
    data = get_classifier_data(sub_num,roi_name, layer_num,schema_train,schema_test)
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
    D['schema_train'] = schema_train
    D['schema_test'] = schema_test
    L.append(D)

## 
results = pd.DataFrame(L)


# In[32]:


results


# In[11]:


Nsubs = len(results.sub_num.unique())
results.to_csv('data/analyses/decodeState_trainView_testRecall-split_by_schema-N%i.csv'%Nsubs)

