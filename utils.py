
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



