
""" loading utilities """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import brainiak
import nilearn as nl
from nilearn import image, plotting, input_data
from scipy.spatial import distance


ALL_SUB_NS = [
  3,4,5,6,7,8,9,10,
  12,13,14,15,17,18,19,
  22,23,24,25,26,27,28,29,
  30,31,32,33,34,35,36,38,
  40,41,42,43,44
  ]
SUB_NS = [30,31,32,33,34,35,36,38]
WED_L = ['W1', 'W2', 'W6', 'W17', 'W19', 'W20', 'W22', 'W23', 'W28', 'W29', 'W34', 'W38']
ROI_NAME_L = [
  'rglasser_AT_net',
  'rglasser_MP_net',
  'rglasser_MTN_net',
  'rglasser_PM_net',
  'rhippocampusAAL',
  'rsherlockAvg_fc_thr5_mpfc',
  # 'rsherlockAvg_fc_thr5_pmc',
  # 'rsherlockAvg_fc_thr5_lTPJ',
  # 'rsherlockAvg_fc_thr5_rTPJ',
  # 'rsherlockAvg_fc_thr5_lSFG',
  # 'rsherlockAvg_fc_thr5_rSFG',
  ]


def load_subj_behav_df(sub_n):
  sub_df = pd.read_csv('behav/from_silvy/recallTranscriptions/S%i.csv'%sub_n,index_col=0).T.fillna(0)
  if sub_n==6: sub_df = sub_df.replace('\n',0)   
  return sub_df

def load_recall_df():
  L = []
  for sub_n in ALL_SUB_NS:
    sub_df = load_subj_behav_df(sub_n) 
    sub_df = pd.melt(sub_df,var_name='wedding', value_name='recall')
    sub_df = sub_df.astype({'recall':int})
    sub_df['subject'] = sub_n
    L.append(sub_df)
  return pd.concat(L)

def load_wed_path_map():
  """ 
  index are subs, cols are wedding numbers
  """
  df = pd.read_pickle("behav/from_silvy/weddOverview_AscendingOrderOfWeddings.pkl")[1:]
  df.index = [i[-1] for i in df.index.str.split('/')]
  df.index = [int(i[0]) for i in df.index.str.split('_')]
  df.columns = [1,2,6,17,19,20,22,23,28,29,34,38]
  return df