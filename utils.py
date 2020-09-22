
""" """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import brainiak
import nilearn as nl
from nilearn import image, plotting, input_data
from scipy.spatial import distance

ALL_SUB_NS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
SUB_NS = np.arange(30,39)
WED_L = ['W1', 'W2', 'W6', 'W17', 'W19', 'W20', 'W22', 'W23', 'W28', 'W29', 'W34', 'W38']
N_WEDDINGS = 12
ROI_NAME_L = [
  'rsherlockAvg_fc_thr5_rTPJ','rsherlockAvg_fc_thr5_pmc',
  'rsherlockAvg_fc_thr5_rSFG','rsherlockAvg_fc_thr5_lSFG',
  'rsherlockAvg_fc_thr5_lTPJ','rsherlockAvg_fc_thr5_mpfc',
  'rhippocampusL_AAL','rhippocampusR_AAL',
  'rhippocampusAAL']

def load_sub4d(sub_n,task='videos',max_len=2000,numpy_output=False):
  """ 
  task = videos / recall2 
  """
  ses = 2
  fpath = 'fmri_data/func/sub-1%i_ses-0%i_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii'%(sub_n,ses,task)
  try:
    img = nl.image.load_img(fpath)
  except:
    print('NOT FOUND:',fpath)
    return None
  img = img.slicer[:,:,:,:max_len] # lengths differ
  if numpy_output:
    img = img.get_fdata() # nilearn into np
  return img

