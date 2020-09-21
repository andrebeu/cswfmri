
""" """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import brainiak
import nilearn as nl
from nilearn import image, plotting, input_data
from scipy.spatial import distance

SUB_NS = np.arange(30,39)
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

