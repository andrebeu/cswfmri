""" """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import brainiak
import nilearn as nl
from nilearn import image, plotting, input_data

from utils import *


for sub_n in SUB_NS:
  print('subj%i'%sub_n)
  # load subj
  sub4d = load_sub4d(sub_n,task='videos',max_len=2000,numpy_output=False)
  for roi_name in ROI_NAME_L:
    print('roi=',roi_name)
    # load mask
    roi_img = nl.image.load_img("fmri_data/rois/%s.nii"%roi_name)
    # plt masks
    plt.figure(figsize=(3,8))
    nl.plotting.plot_glass_brain(roi_img)
    plt.savefig('figures/masks/%s'%roi_name)
    plt.close('all')
    # init & apply mask
    nifti_masker = nl.input_data.NiftiMasker(mask_img=roi_img,high_pass=1/128,t_r=1.5)
    sub4d_masked = nifti_masker.fit_transform(sub4d)
    # save
    save_fpath = "sub-%i_%s"%(sub_n,roi_name)
    np.save('fmri_data/masked/%s'%save_fpath,sub4d_masked)
  

