
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:23:15 2020

@author: melanie
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import *
from cmath import phase
from numpy import array
from scipy.sparse import csr_matrix
import urllib
import glob
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
from random import randint
import math
from circ_stats import *

##################################################################################################
#                                               FUNCTIONS                                        #
##################################################################################################


def decode_continuous_leave1out(dataframe,y, mode):
    # y is fixed only X changes based on time in trial

    acc_prev = []
    std_prev = []
    acc_curr = []
    std_curr = []
    for delta_t_train in range(len(dataframe['bin_sp_prev'][dataframe['bin_sp_prev'].index[0]])):# for each period ['targ_on', 'targ_off', 'go_cue', 'saccade', 'reward']
        # create training dataset: columns=neurons, rows=trials for previous/current trials
        X_prev = pd.DataFrame([dataframe['bin_sp_prev'][n][delta_t_train] for n in dataframe['bin_sp_prev'].index])
        X_curr = pd.DataFrame([dataframe['bin_sp_curr'][n][delta_t_train] for n in dataframe['bin_sp_curr'].index])
        
        # Crossvalidation
        acc_crosscorr_prev=[]
        acc_crosscorr_curr=[]
            
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(X_prev):
            X_train_prev, X_test_prev = X_prev.loc[train_idx], X_prev.loc[test_idx]
            X_train_curr, X_test_curr = X_curr.loc[train_idx], X_curr.loc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # determine labels (sine, cosine)
            y_train_cos = np.cos(y_train)
            y_train_sin = np.sin(y_train)
            #y_test_cos = np.cos(y_test)

            # make linear regression fit for sine/cosine for prev/current trial
            model_prev_cos = LinearRegression().fit(X_train_prev, y_train_cos)
            model_prev_sin = LinearRegression().fit(X_train_prev, y_train_sin)
            model_curr_cos = LinearRegression().fit(X_train_curr, y_train_cos)
            model_curr_sin = LinearRegression().fit(X_train_curr, y_train_sin)

            # make predictions of models
            preds_prev_cos = model_prev_cos.predict(X_test_prev)
            preds_prev_sin = model_prev_sin.predict(X_test_prev)
            preds_curr_cos = model_curr_cos.predict(X_test_curr)
            preds_curr_sin = model_curr_sin.predict(X_test_curr)

            preds_prev = [math.atan2(preds_prev_sin[n],preds_prev_cos[n]) for n in range(len(preds_prev_sin))]
            preds_curr = [math.atan2(preds_curr_sin[n],preds_curr_cos[n]) for n in range(len(preds_prev_sin))]

            # R squared value
            if mode == 'r2':
                acc_crosscorr_prev.append(metrics.r2_score(preds_prev, y_test))
                acc_crosscorr_curr.append(metrics.r2_score(preds_curr, y_test))
            elif mode == 'MSE':
                #acc_crosscorr_prev.append(np.mean(abs(circdist(preds_prev, y_test))))
                #acc_crosscorr_curr.append(np.mean(abs(circdist(preds_curr, y_test))))
                acc_crosscorr_prev.append(np.mean(circdist(preds_prev, y_test)**2))
                acc_crosscorr_curr.append(np.mean(circdist(preds_curr, y_test)**2))
            else:
                print('Mode needs to be either:\n'+'r2 for evaluation with R² metric \n'+\
                          'or\nMSE for evaluation with mean-squared error.')
                return
            

        acc_prev.append(acc_crosscorr_prev)
        acc_curr.append(acc_crosscorr_curr)
    return acc_prev, acc_curr


##################################################################################################
#                                               LOAD DATA                                        #
##################################################################################################

#!wget https://github.com/MelanieTschiersch/SmithData/blob/main/Sa191226.mat?raw=true # , Pe180728.mat, Wa180222.mat
#!mv Sa191226.mat?raw=true Sa191226.mat #, Pe180728.mat, Sa191226.mat

#!wget https://github.com/comptelab/distributedWM/blob/main/smith/PFC_PFC/Sa191202.mat # , Pe180728.mat, Wa180222.mat
#!mv Sa191226.mat?raw=true Sa191226.mat #, Pe180728.mat, Sa191226.mat
data={'Sa': [], 'Pe':[], 'Wa':[]}
for m in ["Sa", "Pe", "Wa"]:
    files = np.sort(glob.glob('../Data/%s*.mat' %m))
    for f in files:
#data = loadmat('distributedWM/smith/PFC_PFC/Sa191202.mat')#Sa191203
        data[m].append(loadmat(f))
        print(f)
    #files_web = np.sort(glob.glob('%s*.mat' %m))
    #for f in files_web:
    #    data[m].append(loadmat(f))#, Pe180728.mat, Wa180222.mat, Sa191226.mat

left_idx = {'Sa': [[] for i in range(len(data['Sa']))], 'Pe':[[] for i in range(len(data['Pe']))], 'Wa':[[] for i in range(len(data['Wa']))]}
right_idx = {'Sa': [[] for i in range(len(data['Sa']))], 'Pe':[[] for i in range(len(data['Pe']))], 'Wa':[[] for i in range(len(data['Wa']))]}
for m in ["Sa", "Pe", "Wa"]:
    for n in range(len(data[m])):
        left_idx[m][n] = data[m][n]['left_idx']
        right_idx[m][n] = data[m][n]['right_idx']
        
dataset=[]
monkey=[]
session=[]
for m in ['Sa','Pe','Wa']:
    for n in range(len(data[m])):
        for line in data[m][n]['dat'][0]:
            dataset.append([row if not isinstance(row, np.ndarray) else row.flat[0] if len(row[0])==1 else row[0] for row in line])
            monkey.append(m)
            session.append(n)


columns = ['trial_id','sp_train', 'outcome', 'timing', 'targ_xy', 'targ_angle', 'saccade_xy', 'saccade_angle']

df_dat = pd.DataFrame(dataset, columns=columns)
df_dat['monkey'] = monkey
df_dat['session'] = session

taskperiods = ['fix', 'targ_on', 'targ_off', 'go_cue', 'saccade', 'reward']
for i in range(len(taskperiods)):
    df_dat[taskperiods[i]] = [line[i].flat[0] for line in df_dat['timing']]
del df_dat['timing']

df_dat['trial_end'] = [df_dat['sp_train'][n].shape[1] for n in range(len(df_dat['sp_train']))]

x_start=[]
x_label=[]
clockw=[]


# CHANGE WM CIRCLE SO THAT 0 ON TOP
help_circle = np.zeros(len(df_dat))
help_circle_sac = np.zeros(len(df_dat))
for idx in range(len(df_dat)):
    if df_dat.targ_angle[idx]<90:
        help_circle[idx] = df_dat.targ_angle[idx]+270
    elif df_dat.targ_angle[idx]==90:
        help_circle[idx] = 0
        help_circle_sac[idx] = 0
    elif df_dat.targ_angle[idx]>90:
        help_circle[idx] = df_dat.targ_angle[idx]-90
    else:
        print('error')
        
    if df_dat.saccade_angle[idx]<90:
        help_circle_sac[idx] = df_dat.saccade_angle[idx]+270
    elif df_dat.saccade_angle[idx]==90:
        help_circle_sac[idx] = 0
    elif df_dat.saccade_angle[idx]>90:
        help_circle_sac[idx] = df_dat.saccade_angle[idx]-90
    else:
        print('error')

df_dat['targ_angle'] = help_circle
df_dat['saccade_angle'] = help_circle_sac


# add value for clockwise, counterclockwise trials
for n in range(len(df_dat['targ_on'])):
    x_start.append([])
    x_label.append([])
    x = csr_matrix(df_dat['sp_train'][n])
    x_start[n] = np.sum(x[:,df_dat['targ_on'][n]:df_dat['targ_off'][n]].toarray(), axis=1)
    x_label[n] = df_dat['targ_angle'][n]
    if circdist(df_dat['targ_angle'][n]*np.pi/180, df_dat['saccade_angle'][n]*np.pi/180)<=0:
        clockw.append('CW') 
    else:
        clockw.append('CCW') 

df_dat['n_cue'] = x_start # all neurons during time of cue
df_dat['clockw'] = clockw
df_dat.head()

##################################################################################################
#                                               SPLIT DATA                                        #
##################################################################################################
 
for mono in ['Sa','Pe','Wa']:
    for sess in range(len(data[mono])):        #only use Sa, sess0
        df_Sa0 = df_dat.loc[(df_dat['monkey']==mono) & (df_dat['session']==sess)].reset_index()
        
        # make spike trains into csr matrix for each trial
        spike_matrix = [csr_matrix(df_Sa0.loc[n,'sp_train']) for n in df_Sa0['sp_train'].index]
        df_Sa0['n_mat'] = spike_matrix
        
        # determine border points between different time periods, until beginning of delay
        bins = 50
        timings = ['fix','targ_on','targ_off', 'go_cue']#,'saccade', 'reward', 'trial_end']# discrete timings
        borders=[]
        for period in range(len(timings)):
            #help += int(min([(df_Sa0.loc[n, timings[period+1]]-df_Sa0.loc[n, timings[period]]) for n in range(len(df_Sa0))])/bins)
            borders.append(int(min(df_dat[timings[period]]/bins)))
        #print(borders)
        
        # determine border points INDIVID trials between different time periods, for end of delay
        timings2 = ['go_cue','saccade', 'reward', 'trial_end']
        t_borders2 = ['delay_start','delay_end','saccade', 'reward', 'trial_end', 'end_start', 'end']#
        borders2={'delay_start': [], 'delay_end': [], 'saccade': [], 'reward':[], 'trial_end':[], 'end_start':[], 'end':[]}##np.zeros((len(timings2)+1, len(df_Sa0)))
        for i,m in enumerate(borders2.keys()):
            if i==0:
                #create shifted "start" of delay
                borders2[m] = ((df_Sa0['go_cue'].values)/bins - min(((df_Sa0['go_cue'].values-df_Sa0['targ_off'].values)/bins))).astype(int)#
            elif i ==1:
                # delay end
                borders2[m] = ((df_Sa0['go_cue'].values)/bins).astype(int)
                #np.array([int(df_Sa0.loc[n,timings2[0]]/bins)-(borders[-1]) for n in range(len(df_Sa0))])
            elif m =='end_start':
                # shifted "start" of trial end : complete end of trial - minimum(trial_end-reward)
                borders2[m] = [int(df_Sa0.loc[n,'trial_end']/bins)-int(min((df_Sa0.loc[:,'trial_end']-df_Sa0.loc[:,'reward'])/bins)) for n in df_Sa0.index]#
            elif m == 'end':
                borders2[m] = [int(df_Sa0.loc[n,'trial_end']/bins) for n in df_Sa0.index]
            else:
            # create end delay, saccade start, reward start, trial_end through using minimum distance between periods, adding to delay_end, saccade_end,..
                borders2[m] = np.array(borders2[t_borders2[i-1]]) + min([int((df_Sa0.loc[n,timings2[i-1]]-df_Sa0.loc[n,timings2[i-2]])/bins) for n in df_Sa0.index])
            #print(min([int((df_Sa0.loc[n,timings2[period]]- df_Sa0.loc[n,timings2[period-1]])/bins) for n in range(len(df_Sa0))]))
            #np.array(min([int((df_Sa0.loc[n,timings2[period]]- df_Sa0.loc[n,timings2[period-1]]))/bins for n in range(len(df_Sa0))]))
        
        # add shift between trial short end and trial long start
        borders.append(borders[-1]+min(np.array(borders2['trial_end'])- np.array(borders2['delay_start'])))
        
        bin_sp_trials=[]
        period_spikes=[]
        for trial in df_Sa0.index:# for all trials
            binned_spikes = []
            for period in range(len(timings[:-1])):# for all discrete timings
                for t in range(borders[period+1]-borders[period]): # for all time bins in discrete timings:           
                    # sum the matrix of neurons at timings in bin
                    binned_spikes.append(np.sum(df_Sa0.loc[trial, 'n_mat'][:,df_Sa0.loc[trial,timings[period]]+t*bins:df_Sa0.loc[trial,timings[period]]+t*bins+bins].toarray(), axis=1))
                #print(t)
            #print(len(binned_spikes[0]))
            bin_sp_trials.append(binned_spikes)
        
        
        bin_sp_trials_pastdelay=[]
        period_spikes=[]
        for idx, trial in enumerate(df_Sa0.index):# for all trials
            binned_spikes = []
            for period in range(len(borders2)-1):# for all discrete timings
                for t in range(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0]): # for number of time bins in discrete timings:           
                    # sum the matrix of neurons at timings in bin
                    binned_spikes.append(np.sum(df_Sa0.loc[trial, 'n_mat'][:,borders2[t_borders2[period]][idx]*bins+t*bins:borders2[t_borders2[period]][idx]*bins+t*bins+bins].toarray(), axis=1))
                #print(t)
            #print(len(binned_spikes[0]))
            bin_sp_trials_pastdelay.append(binned_spikes)
        
        bin_sp_complete = np.append(bin_sp_trials,bin_sp_trials_pastdelay, axis=1)
        
        borders_full=[]
        borders_full = np.append(borders,borders[-1]+borders2['delay_end'][0]-borders2['delay_start'][0])
        borders_full = np.append(borders_full,borders_full[-1]+borders2['saccade'][0]-borders2['delay_end'][0])
        borders_full = np.append(borders_full,borders_full[-1]+borders2['reward'][0]-borders2['saccade'][0])
        borders_full = np.append(borders_full,borders_full[-1]+borders2['trial_end'][0]-borders2['reward'][0])
        borders_full = np.append(borders_full,borders_full[-1]+borders2['end'][0]-borders2['trial_end'][0])
        
        borders_pastdelay = borders_full[len(borders):]
        
        ##################################################################################################
        #                                          SERIAL BIAS                                           #
        ##################################################################################################
        
        serial = {'trial_id':[], 'target_prev': [], 'targ_off_prev':[], 'go_cue_prev':[], 'response_prev': [],\
                  'delay_prev': [],'bin_sp_prev':[], 'target_curr': [], 'targ_on_curr':[], 'response_curr': [],\
                  'delay_curr': [], 'bin_sp_curr':[],  'monkey': [], 'cw':[]}
        for trial,idx in enumerate(df_Sa0.index[:-1]):
            if ((df_Sa0['trial_id'][idx]+1) == (df_Sa0['trial_id'][idx+1])):
                serial['trial_id'].append(idx)
                serial['target_prev'].append(df_Sa0['targ_angle'][idx]*np.pi/180)
                serial['targ_off_prev'].append(round(df_Sa0['targ_off'][idx], 5))
                serial['go_cue_prev'].append(df_Sa0['go_cue'][idx])
                serial['response_prev'].append(round(df_Sa0['saccade_angle'][idx]*np.pi/180, 5))
                serial['delay_prev'].append(df_Sa0['go_cue'][idx]-df_dat['targ_off'][idx])
                serial['bin_sp_prev'].append(bin_sp_complete[trial])
                serial['target_curr'].append(df_Sa0['targ_angle'][idx+1]*np.pi/180)
                serial['targ_on_curr'].append(df_Sa0['targ_on'][idx+1])
                serial['response_curr'].append(round(df_Sa0['saccade_angle'][idx+1]*np.pi/180, 5))
                serial['delay_curr'].append(df_Sa0['go_cue'][idx+1]-df_Sa0['targ_off'][idx+1]) 
                serial['bin_sp_curr'].append(bin_sp_complete[trial+1])
                serial['monkey'].append(df_Sa0['monkey'][idx])
                serial['cw'].append((df_Sa0['clockw'][idx]=='CW'))
                
        df_serial = pd.DataFrame(serial)
        
        
        ##################################################################################################
        #                                SPLIT IN CW VS CCW TRIALS                                       #
        ##################################################################################################
        
        mode='MSE'
        
        y = df_serial['target_prev'].values-np.pi
        
        acc_prev, acc_curr = decode_continuous_leave1out(df_serial, y, mode)
        
        acc_prev_cw = np.array(acc_prev)[np.where(df_serial['cw'])]
        acc_curr_cw = np.array(acc_curr)[np.where(df_serial['cw'])]
        acc_prev_ccw = np.array(acc_prev)[np.where(df_serial['cw']==False)]
        acc_curr_ccw = np.array(acc_curr)[np.where(df_serial['cw']==False)]
        
        file1 = open("CWvsCCW_AllMonkeys.txt", "a+")
        str_dictionary = repr(acc_prev_cw)
        file1.write(str_dictionary + "\n")
        str_dictionary = repr(acc_curr_cw)
        file1.write(str_dictionary + "\n")
        str_dictionary = repr(acc_prev_ccw)
        file1.write(str_dictionary + "\n")
        str_dictionary = repr(acc_curr_ccw)
        file1.write(str_dictionary + "\n")
        file1.close()
        
        print('saved monkey '+str(m)+str(n))