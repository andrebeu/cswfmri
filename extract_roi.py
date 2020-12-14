#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nilearn as nl
from nilearn import image, plotting, input_data
import itertools


# In[2]:


def load_sub4d(sub_n,task='videos',max_len=4000,numpy_output=False):
  """
  task = videos / recall2 
  """
  ses = 2
  data_dir = 'data/fmri/wholebrain/'
  fpath = data_dir + 'sub-1%.2i_ses-0%i_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'%(sub_n,ses,task)
  try:
    img = nl.image.load_img(fpath)
  except:
    print('NOT FOUND:',fpath)
    return None
  img = img.slicer[:,:,:,:max_len] # lengths differ
  if numpy_output:
    img = img.get_fdata() # nilearn into np
  return img


# In[3]:


roi_name_L = [
  'SnPM_filtered_FDR',
  'rglasser_AT_net',
  'rglasser_MP_net',
  'rglasser_MTN_net',
  'rglasser_PM_net',
  'rhippocampusAAL', 
]


# # main loop

# In[4]:


sub_ns = np.arange(45)
filter_denom_L = [None,128,480] 
motioncorr_L = [True, False]


# In[5]:


for sub_n,task in itertools.product(sub_ns,['videos','recall2']):
  # load wholebrain data
  sub4d = load_sub4d(sub_n,task=task,max_len=4000,numpy_output=False)
  if sub4d == None: 
    print('wholebrain data for sub',sub_n,task,'not found')
    continue
  for roi_name,filter_denom,motioncorr in itertools.product(roi_name_L,filter_denom_L,motioncorr_L):
    print('subj%i'%sub_n,'roi=',roi_name,'filter',filter_denom,'motion',motioncorr)
    # load & threshold mask
    try: 
      roi_img = nl.image.load_img("data/fmri/rois/%s.nii"%roi_name)
    except: 
      print('roi not found',roi_name)
      continue
    # thresholding: functional masks are different
    if roi_name=='SnPM_filtered_FDR':
      roi_img = nl.image.math_img('img>0',img=roi_img)
    else:
      roi_img = nl.image.threshold_img(roi_img,0.5)
    # init masker
    if filter_denom:
      nifti_masker = nl.input_data.NiftiMasker(mask_img=roi_img,high_pass=1/filter_denom,t_r=1.5)
    else:
      nifti_masker = nl.input_data.NiftiMasker(mask_img=roi_img)
    try:
      # apply mask and regress motion
      if motioncorr:
        sub4d_masked = nifti_masker.fit_transform(sub4d,
          confounds="data/fmri/selected_nuisance/sub-%i_ses-02_task-%s_confounds_selected.txt"%(100+sub_n,task[:6]))
      else:
        sub4d_masked = nifti_masker.fit_transform(sub4d)
    except:
      print('- err running nifti_masker.fit_transform on',sub_n,roi_name)
      continue
    # save
    save_fpath = "sub-%i_task-%s_roi-%s-filter_%s-motioncorr_%s"%(sub_n,task,roi_name,str(filter_denom),motioncorr)
    np.save('data/fmri/roi_act/%s'%save_fpath,sub4d_masked)


# In[ ]:




