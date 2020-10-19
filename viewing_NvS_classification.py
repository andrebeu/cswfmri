import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

import sklearn
import brainiak
import nilearn as nl
from nilearn import image, plotting, input_data

from scipy.spatial import distance

from utils import *

from sklearn.linear_model import LogisticRegression


# # load dataframes with timing and order information

""" onset and offset TR for each state of each wedding """
timing_df = pd.read_csv('behav/exp_timing.csv',index_col=0)
order_df = pd.read_csv('behav/subject_wedding_order.csv',index_col=0)


def extract_viewing_TRs(wed_num):
  """
  returns a L containing the valid TRs in wed_num
    NB consistent across subjects
  valid TRs are those within a vid
  """
  wed_timing_df = timing_df[timing_df.wed_num == wed_num]
  L = []
  for i,row in wed_timing_df.iterrows():
    vid_TRs = np.arange(row.onset_TR,row.offset_TR)
    L.extend(vid_TRs)
  return L


get_wed_label = lambda sub_num,wed_num: order_df[(
    order_df.sub_num == sub_num) & (
    order_df.wed_num == wed_num
  )].NorS.values[0]


def get_fold_info(sub_num,fold_L):
  """ 
  """
  Y_label = []
  X_TRs = []
  ## TEST TRS
  for wed_num in fold_L:
    # X TRs
    wed_TRs = extract_viewing_TRs(wed_num)
    X_TRs.extend(wed_TRs)
    # Y labels
    y_wed = np.repeat(get_wed_label(sub_num,wed_num)=='N',len(wed_TRs)).astype(int)
    Y_label.extend(y_wed)
  return X_TRs,Y_label


def load_sub_roi(sub_num,task,roi_name):
  fpath = "sub-%i_task-%s_roi-%s.npy" %(sub_num,task,roi_name)
  return np.load('fmri_data/masked/'+fpath)


fold_full_L = [
    [0,1],[2,3],[4,5],
    [6,7],[8,9],[10,11]
  ]


get_fold_L_train = lambda fold_num: [j for i in fold_full_L if i!=fold_full_L[fold_num] for j in i]


# xval loop


""" 
xval loop 
"""
clf_c = 0.001
roi_name = 'rglasser_PM_net'
yhat_L = []
for sub_num in SUB_NS:
  print(sub_num)
  sub_roi = load_sub_roi(sub_num,'videos',roi_name)
  # fold information
  for fold_num in range(6):
    fold_L_test = fold_full_L[fold_num]
    fold_L_train = get_fold_L_train(fold_num)
    # classifier init
    clf = sklearn.linear_model.LogisticRegression(solver='liblinear',C=clf_c)
    # TRAIN
    X_TRs_train,Y_train = get_fold_info(sub_num,fold_L_train)
    X_train = sub_roi[X_TRs_train,:]
    clf.fit(X_train,Y_train)
    # TEST
    X_TRs_test,Y_test = get_fold_info(sub_num,fold_L_test)
    X_test = sub_roi[X_TRs_test,:]
    yhat_full = clf.predict_proba(X_test)
    # split eval of test TRs into two weddings
    # and plot proba of correct schema
    yhat_first = yhat_full[:int(len(yhat_full)/2),Y_test[0]]
    yhat_second = yhat_full[int(len(yhat_full)/2):,Y_test[-1]]
    yhat_L.append(yhat_first)
    yhat_L.append(yhat_second)


# In[14]:


## compute mean
yhat = np.array(yhat_L)
yhat.shape
M = np.array(yhat_L).mean(0)
S = np.array(yhat_L).std(0) / np.sqrt(len(yhat_L))


# In[20]:


## save classification mean accuracy
fpath = 'analyses/NvS_viewing_classification/Macc-%s'%(roi_name)
np.save(fpath,M)


# In[15]:


## plt
ax = plt.gca()
ax.plot(M)
ax.fill_between(np.arange(len(M)),M-S,M+S,alpha=.3)
ax.axhline(0.5,c='k',ls='--')
ax.set_ylim(0.2,0.8)
ax.set_title('roi-%s_C_%.4f'%(roi_name,clf_c))
plt.savefig('figures/NvS_logreg_roi-%s_C%.4f.jpg'%(roi_name,clf_c))


# In[ ]:




