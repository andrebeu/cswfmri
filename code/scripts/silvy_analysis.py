


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import csv
import numpy as np
from numpy import array
from scipy.sparse import csr_matrix
from ast import literal_eval
from collections import Counter
from scipy import sparse
import glob
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from scipy.sparse import vstack
from scipy.sparse import hstack
import gc
import matplotlib.pyplot as plt
from scipy import stats
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit

###

# load confounds file to regress out confounds (motion, csf, wm)
fname ="%s_%s_task-%s_confounds_selected.txt" % (sub,ses,task[currTask])
confoundsname = nuisance_dir + fname
confound_file = np.loadtxt(confoundsname)
print(fname)
            
# Apply mask and create maskedData (= TR x voxel matrix)
nifti_masker = NiftiMasker(mask_img=final_mask,  high_pass=1/128, t_r=1.5)
maskedData = nifti_masker.fit_transform(epi_data, confounds=confound_file)

"""
What will also be at some point relevant which is 
how it loads the timeseries nifti files from which 
it extracts the data for each subject and each relevant 
task (in your case only 1, the recall) within a loop: """

def load_epi_data(sub, ses, task):
    # Load MRI file (in Nifti format)
    epi_in = os.path.join(
      data_dir, sub, ses, 'func', 
      "%s_%s_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % (sub, ses, task)
      )
    epi_data = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data


###


# settings
saveEachWeddSeparately = 1
VersionEvents = 1 #1 is campfire flower, 2 is coin tech, 3 is egg painting

output_dir = '/jukebox/norman/scollin/schema/data/bids/Norman/Silvy/schema/derivatives/results_classifier/TRbyTR/LogisticRegression/event/'
onsettimes_dir = '/jukebox/norman/scollin/schema/data/bids/Norman/Silvy/schema/derivatives/extractedWeddingTRs_MNIspace/'

AllSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]

ROI_name = 'eventClassifierFDR05_leaveOneOutMasks'

timeseries_dir = '/jukebox/norman/scollin/schema/data/bids/Norman/Silvy/schema/derivatives/extractedWeddingTRs_MNIspace/'
TRlength = 74
TRlimit = 80
NrWedds = 12
Sample_Length = TRlength * NrWedds
EndIntro = 17
EndEvent1 = 23
EndEvent2 = 35
EndEvent3 = 50
EndEvent4 = 66

## load pickle with matrix subj x wedd, which path is each wedding for each subj, ascending order of weddings based on index
directory = '/jukebox/norman/scollin/schema/data/bids/Norman/Silvy/schema/derivatives/resultsHMM/'
dfpathAB_transposed = pd.read_pickle(directory + 'weddOverview_AscendingOrderOfWeddings.pkl')
newpathAB = dfpathAB_transposed.drop(dfpathAB_transposed.index[0])
newpathAB.index = range(len(newpathAB))

# make sure there is ascending order of wedding timeseries to be loaded later on
FileOrderCorrected = [0,3,11,1,2,4,5,6,7,8,9,10]


TotalDataAllWeddsGroup = pd.DataFrame(columns=[0])
accuracyGroup = []
outputMatrix_timeline = pd.DataFrame(columns=[])

# loop over subjs
for CurrSub in range(len(AllSubjects)):
    
    TotalDataAllWedds = pd.DataFrame(columns=[0])
    
    print('run subject',AllSubjects[CurrSub])
    
    # what is A and what is B path for this subj
    currRow = newpathAB.loc[CurrSub]
    listCurrRow = currRow.values.tolist()
    nplistCurrRow = np.array(listCurrRow)
    
    # load files with wedding timeseries
    path = timeseries_dir + str(AllSubjects[CurrSub]) + '/' + ROI_name + '/'
    files = [f for f in glob.glob(path + "*videos1_%s_*.npy" %(ROI_name), recursive=True)]
    
    # sort files into sensible order (same across subjects)
    files = sorted(files)
        
    counter = 0
    counterNA = 0
    counterNB = 0
    counterSA = 0
    counterSB = 0

    # loop over weddings
    for currWedd in FileOrderCorrected:
        
        timeseries = np.load(files[currWedd])
        print(files[currWedd])

        # make sure to remove TRs where people are answering the 2AFC questions (and only keep timeseries of actual watching of videos)
        if len(timeseries) > TRlimit:
            timeseries = timeseries[list(chain(
              range(0,34),range(37,52),range(55,len(timeseries)))
            )]
            print(len(timeseries))

        splittedName = np.char.split(files[currWedd], sep = '/')
        splittedName = splittedName.tolist()
        finalName = float(splittedName[-1][4:6])

        fileName = files[currWedd].split('/')

        if VersionEvents is 1:
        # only campfire/flower TRs
            TotalData = pd.DataFrame(timeseries[EndEvent1:EndEvent2])
            print('campfire flower')

            if nplistCurrRow[counter] == 'NA':
                TotalData['weddIdx'] = 1 #true label (corresponding to event)
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData))) #TR
                TotalData['counter'] = counterNA # counter to later create predefined split
                TotalData['WhichWedd'] = float(fileName[14][4:7]) #which wedding is it
                counterNA += 1
            elif nplistCurrRow[counter] == 'SA':
                TotalData['weddIdx'] = 1
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterSA
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterSA += 1
            elif nplistCurrRow[counter] == 'NB':
                TotalData['weddIdx'] = 2
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterNB
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterNB += 1
            elif nplistCurrRow[counter] == 'SB':
                TotalData['weddIdx'] = 2
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterSB
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterSB += 1

        elif VersionEvents is 2:
        # only coin/torch TRs
            TotalData = pd.DataFrame(timeseries[EndEvent2:EndEvent3])
            print('coin torch')

            if nplistCurrRow[counter] == 'NA':
                TotalData['weddIdx'] = 1
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterNA
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterNA += 1
            elif nplistCurrRow[counter] == 'SA':
                TotalData['weddIdx'] = 2
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterSA
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterSA += 1
            elif nplistCurrRow[counter] == 'NB':
                TotalData['weddIdx'] = 2
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterNB
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterNB += 1
            elif nplistCurrRow[counter] == 'SB':
                TotalData['weddIdx'] = 1
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterSB
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterSB += 1

        elif VersionEvents is 3:
            TotalData = pd.DataFrame(timeseries[EndEvent3:EndEvent4])
            print('egg painting')

            if nplistCurrRow[counter] == 'NA':
                TotalData['weddIdx'] = 1
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterNA
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterNA += 1
            elif nplistCurrRow[counter] == 'SA':
                TotalData['weddIdx'] = 1
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterSA
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterSA += 1
            elif nplistCurrRow[counter] == 'NB':
                TotalData['weddIdx'] = 2
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterNB
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterNB += 1
            elif nplistCurrRow[counter] == 'SB':
                TotalData['weddIdx'] = 2
                TotalData['subj'] = CurrSub
                TotalData['TRidx'] = list(range(len(TotalData)))
                TotalData['counter'] = counterSB
                TotalData['WhichWedd'] = float(fileName[14][4:7])
                counterSB += 1

        print(nplistCurrRow[counter])
        
        TotalDataAllWedds = TotalDataAllWedds.append(TotalData,sort=True)
        
        counter += 1
        
    ps = PredefinedSplit(np.array(TotalDataAllWedds['counter']))

    print(np.array(TotalDataAllWedds['counter']))
    print(ps.get_n_splits())

    OutputCurrSubjTotal = pd.DataFrame(columns=[])
    OutputCurrSubjTotal2 = pd.DataFrame(columns=[])

    # set X and y
    y = TotalDataAllWedds[['weddIdx']].values.ravel()
    print(len(y))

    TRidxToSave = TotalDataAllWedds[['TRidx']].values.ravel()
    WEDDidxToSave = TotalDataAllWedds[['WhichWedd']].values.ravel()

    X = np.array(TotalDataAllWedds.drop(columns=['subj','weddIdx','TRidx','counter','WhichWedd']))
    print(X.shape)

    accuracy = []
    #accuracy = pd.DataFrame(columns=[])

    # run classifier
    for train_index, test_index in ps.split():
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        TRidxToSave_train, TRidxToSave_test = TRidxToSave[train_index], TRidxToSave[test_index]
        WEDDidxToSave_train, WEDDidxToSave_test = WEDDidxToSave[train_index], WEDDidxToSave[test_index]
        
        print(len(X_train))
        print(len(X_test))
        print(len(y_train))
        print(len(y_test))

        clf = LogisticRegression(solver='liblinear').fit(X_train, y_train)
        # calculate output probabilities
        probabilityOutputPD = pd.DataFrame(clf.predict_proba(X_test))

        # add true label to each sample
        probabilityOutputPD['weddIdx'] = np.array(y_test)
        probabilityOutputPD['TRidx'] = np.array(TRidxToSave_test)
        probabilityOutputPD['currWeddIdx'] = np.array(WEDDidxToSave_test)
        
        probabilityOutputPD['proba_correct'] = pd.DataFrame(np.zeros(len(probabilityOutputPD)))
        
        for currSample in range(len(probabilityOutputPD)):
        
            currSampleAll = probabilityOutputPD.iloc[currSample]
    
            if currSampleAll['weddIdx'] == 1:
                probabilityOutputPD['proba_correct'][currSample] = currSampleAll[0]
            elif currSampleAll['weddIdx'] == 2:
                probabilityOutputPD['proba_correct'][currSample] = currSampleAll[1]        
        
        accuracy.append(probabilityOutputPD['proba_correct'].mean())

        print(accuracy)

        Results_timeline = probabilityOutputPD.sort_values(by=['TRidx'])
        timelineTotal = Results_timeline.groupby('TRidx').mean()
        OutputCurrSubj = timelineTotal['proba_correct']
        OutputCurrSubjTotal = OutputCurrSubjTotal.append(OutputCurrSubj)
        print(OutputCurrSubjTotal)
        print(OutputCurrSubjTotal.mean(axis=0))

        MeanTmp2 = probabilityOutputPD.sort_values(by=['currWeddIdx','TRidx'])
        OutputCurrSubjTotal2 = OutputCurrSubjTotal2.append(MeanTmp2)

    if saveEachWeddSeparately is 1:
        if VersionEvents is 1:
            OutputCurrSubjTotal2[['currWeddIdx','TRidx','proba_correct']].to_csv(output_dir + 'indiv_subj_resultsFiles/saveWholeTimelineOfEachWedd/' + ROI_name + '_s' + str(AllSubjects[CurrSub]) + '_TimeLines_WithinSubj_campfireFlower.csv')
            print('campfire flower')
        elif VersionEvents is 2:
            OutputCurrSubjTotal2[['currWeddIdx','TRidx','proba_correct']].to_csv(output_dir + 'indiv_subj_resultsFiles/saveWholeTimelineOfEachWedd/' + ROI_name + '_s' + str(AllSubjects[CurrSub]) + '_TimeLines_WithinSubj_coinTorch.csv')
            print('coin torch')
        elif VersionEvents is 3:
            OutputCurrSubjTotal2[['currWeddIdx','TRidx','proba_correct']].to_csv(output_dir + 'indiv_subj_resultsFiles/saveWholeTimelineOfEachWedd/' + ROI_name + '_s' + str(AllSubjects[CurrSub]) + '_TimeLines_WithinSubj_eggPainting.csv')
            print('egg painting')
    
    
    outputMatrix_timeline = outputMatrix_timeline.append(pd.Series(OutputCurrSubjTotal.mean(axis=0),name=str(CurrSub)))


    accuracy = np.array(accuracy).mean()
    accuracyGroup.append(accuracy)
    print(accuracyGroup)


