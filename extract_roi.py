#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nilearn as nl
from nilearn import image, plotting, input_data


# sub-102_ses-02_task-recall_confounds_selected.txt are the ones for sub2 for the recall task. 
# 
# pass as argument to NiftiMasker when extract the data from the ROI-mask from the images

# In[2]:


def load_sub4d(sub_n,task='videos',max_len=4000,numpy_output=False):
  """
  task = videos / recall2 
  """
  ses = 2
  fpath = 'data/fmri/func/sub-1%.2i_ses-0%i_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'%(sub_n,ses,task)
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


ROI_NAME_L = [
  'SnPM_filtered_FDR',
  'rglasser_AT_net',
  'rglasser_MP_net',
  'rglasser_MTN_net',
  'rglasser_PM_net',
  'rhippocampusAAL', 
]


# # main loop

# In[4]:


debug=False
if debug:
  sub4d = load_sub4d(33,task=task,max_len=4000,numpy_output=False)


# In[ ]:


for sub_n in np.arange(45):
  # load subj
  for task in ['videos','recall2']:
    if not debug:
      sub4d = load_sub4d(sub_n,task=task,max_len=4000,numpy_output=False)
    if sub4d == None: continue
    for roi_name in ROI_NAME_L:
      print('subj%i'%sub_n,'roi=',roi_name)
      # load & threshold mask
      try:
        roi_img = nl.image.load_img("data/fmri/rois/%s.nii"%roi_name)
      except:
        print('roi not found',roi_name)
        continue
      if roi_name=='SnPM_filtered_FDR':
        # functional ROI
        roi_img = nl.image.math_img('img>0',img=roi_img)
      else:
        roi_img = nl.image.threshold_img(roi_img,0.5)
      # plt masks
      plt.figure(figsize=(3,8))
      nl.plotting.plot_glass_brain(roi_img)
      plt.savefig('figures/masks/%s'%roi_name)
      plt.close('all')
      # init & apply mask
      nifti_masker = nl.input_data.NiftiMasker(mask_img=roi_img,high_pass=1/128,t_r=1.5)
      try:
        sub4d_masked = nifti_masker.fit_transform(sub4d,
                        confounds="data/fmri/selected_nuisance/sub-%i_ses-02_task-%s_confounds_selected.txt"%(
                          100+sub_n,task[:6]))
      except:
        print('err masking S',sub_n,roi_name)
        continue
      # save
      save_fpath = "sub-%i_task-%s_roi-%s"%(sub_n,task,roi_name)
      np.save('data/fmri/masked/%s'%save_fpath,sub4d_masked)

