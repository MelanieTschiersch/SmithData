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
from sklearn import metrics
from random import randint
import math
from circ_stats import *

##################################################################################################
#                                               FUNCTIONS                                        #
##################################################################################################


def decode_continuous_prev_cur(dataframe):
    # y is fixed only X changes based on time in trial
    y = dataframe['target_prev'].values*np.pi/180-np.pi# target_prev # response_prev


    mse_acc_prev = []
    mse_std_prev = []
    mse_acc_curr = []
    mse_std_curr = []
    for delta_t_train in range(len(df_serial['bin_sp_prev'][df_serial['bin_sp_prev'].index[0]])):# for each period ['targ_on', 'targ_off', 'go_cue', 'saccade', 'reward']
        # create training dataset: columns=neurons, rows=trials for previous/current trials
        X_prev = pd.DataFrame([dataframe['bin_sp_prev'][n][delta_t_train] for n in dataframe['bin_sp_prev'].index])
        X_curr = pd.DataFrame([dataframe['bin_sp_curr'][n][delta_t_train] for n in dataframe['bin_sp_curr'].index])
        
        # Crossvalidation
        mse_crosscorr_prev=[]
        mse_crosscorr_curr=[]
        for k in range(0,10):# k=10 as in Barbosa2020
            #train test split
            rand_state = randint(0,1000)# to get the same split for previous, current
            X_train_prev ,X_test_prev ,y_train, y_test = train_test_split(X_prev, y, test_size = 0.20, random_state = rand_state)
            X_train_curr ,X_test_curr ,y_train, y_test = train_test_split(X_curr, y, test_size = 0.20, random_state = rand_state)
            
    
            # determine labels (sine, cosine)
            y_train_cos = np.cos(y_train)
            y_train_sin = np.sin(y_train)
            
            # randomly shuffle train and test data
            #X_train_prev = X_train_prev.sample(frac=1, axis=1).reset_index(drop=True)
            #X_test_prev = X_test_prev.sample(frac=1, axis=1).reset_index(drop=True)
            #X_train_curr = X_train_curr.sample(frac=1, axis=1).reset_index(drop=True)
            #X_test_curr = X_test_curr.sample(frac=1, axis=1).reset_index(drop=True)
            #np.random.shuffle(y_train_cos)
            #np.random.shuffle(y_train_sin)
            #np.random.shuffle(y_test)

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

            #preds_prev_cos[np.where(preds_prev_cos>1)]=1
            #preds_prev_cos[np.where(preds_prev_cos<-1)]=-1
            #preds_prev_sin[np.where(preds_prev_sin>1)]=1
            #preds_prev_sin[np.where(preds_prev_sin<-1)]=-1

            preds_prev = [math.atan2(preds_prev_sin[n],preds_prev_cos[n]) for n in range(len(preds_prev_sin))]
            preds_curr = [math.atan2(preds_curr_sin[n],preds_curr_cos[n]) for n in range(len(preds_prev_sin))]

            # MSE
            mse_crosscorr_prev.append(np.mean(circdist(preds_prev, y_test)**2))
            mse_crosscorr_curr.append(np.mean(circdist(preds_curr, y_test)**2))
            
        
        mse_acc_prev.append(np.mean(mse_crosscorr_prev))
        mse_std_prev.append(np.std(mse_crosscorr_prev))
        mse_acc_curr.append(np.mean(mse_crosscorr_curr))
        mse_std_curr.append(np.std(mse_crosscorr_curr))
    return mse_acc_prev, mse_std_prev, mse_acc_curr, mse_std_curr

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



##################################################################################################
#                                               SPLIT DATA                                        #
##################################################################################################
 
for mono in ['Sa','Pe','Wa']:# for each monkey
    for sess in range(len(data[mono])):#for each session
    #only use Sa, sess0
        df_Sa0 = df_dat.loc[(df_dat['monkey']==mono) & (df_dat['session']==sess)]
        
        # make spike trains into csr matrix for each trial
        spike_mat = [csr_matrix(df_Sa0.loc[n,'sp_train']) for n in df_Sa0['sp_train'].index]
        df_Sa0['n_mat'] = spike_mat
        
        # determine border points between different time periods, until beginning of delay
        bins = 50
        timings = ['fix','targ_on','targ_off', 'go_cue']#,'saccade', 'reward', 'trial_end']# discrete timings
        help = 0
        borders=[]
        for period in range(len(timings)):
            #help += int(min([(df_Sa0.loc[n, timings[period+1]]-df_Sa0.loc[n, timings[period]]) for n in range(len(df_Sa0))])/bins)
            #borders.append(help)
            borders.append(int(min(df_dat[timings[period]]/bins)))
        #print(borders)
        
        # determine border points INDIVID trials between different time periods, for end of delay
        timings2 = ['go_cue','saccade', 'reward', 'trial_end']
        t_borders2 = ['delay_start','delay_end','saccade', 'reward', 'trial_end']
        borders2={'delay_start': [], 'delay_end': [], 'saccade': [], 'reward':[], 'trial_end':[]}#np.zeros((len(timings2)+1, len(df_Sa0)))
        for i,m in enumerate(borders2.keys()):
            if i==0:
                #create shifted "start" of delay
                borders2[m] = [int(df_Sa0.loc[n,timings2[0]]/bins)-(borders[-1]-np.sum(borders[:-1])) for n in df_Sa0.index]#
            elif i ==1:
                # delay end
                borders2[m] = [int(df_Sa0.loc[n,timings2[0]]/bins) for n in df_Sa0.index]
                #np.array([int(df_Sa0.loc[n,timings2[0]]/bins)-(borders[-1]) for n in range(len(df_Sa0))])
            else:
            # create end delay, saccade start, reward start, trial_end through using minimum distance between periods, adding to delay_end, saccade_end,..
                borders2[m] = np.array(borders2[t_borders2[i-1]]) + min([int((df_Sa0.loc[n,timings2[i-1]]-df_Sa0.loc[n,timings2[i-2]])/bins) for n in df_Sa0.index])
            #print(min([int((df_Sa0.loc[n,timings2[period]]- df_Sa0.loc[n,timings2[period-1]])/bins) for n in range(len(df_Sa0))]))
            #np.array(min([int((df_Sa0.loc[n,timings2[period]]- df_Sa0.loc[n,timings2[period-1]]))/bins for n in range(len(df_Sa0))]))

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
        
        borders_pastdelay = borders_full[len(borders):]
        
        ##################################################################################################
        #                                          SERIAL BIAS                                           #
        ##################################################################################################
        
        serial = {'trial_id':[], 'target_prev': [], 'targ_off_prev':[], 'go_cue_prev':[], 'response_prev': [], 'delay_prev': [],'bin_sp_prev':[], 'target_curr': [], 'targ_on_curr':[], 'response_curr': [], 'delay_curr': [], 'bin_sp_curr':[],  'monkey': []}
        for trial,idx in enumerate(df_Sa0.index[:-1]):
            if ((df_Sa0['trial_id'][idx]+1) == (df_Sa0['trial_id'][idx+1])):
                serial['trial_id'].append(idx)
                serial['target_prev'].append(df_Sa0['targ_angle'][idx])
                serial['targ_off_prev'].append(round(df_Sa0['targ_off'][idx]*np.pi/180, 5))
                serial['go_cue_prev'].append(df_Sa0['go_cue'][idx])
                serial['response_prev'].append(round(df_Sa0['saccade_angle'][idx]*np.pi/180, 5))
                serial['delay_prev'].append(df_Sa0['go_cue'][idx]-df_dat['targ_off'][idx])
                serial['bin_sp_prev'].append(bin_sp_complete[trial])
                serial['target_curr'].append(df_Sa0['targ_angle'][idx+1])
                serial['targ_on_curr'].append(df_Sa0['targ_on'][idx+1])
                serial['response_curr'].append(round(df_Sa0['saccade_angle'][idx+1]*np.pi/180, 5))
                serial['delay_curr'].append(df_Sa0['go_cue'][idx+1]-df_Sa0['targ_off'][idx+1]) 
                serial['bin_sp_curr'].append(bin_sp_complete[trial+1])
                serial['monkey'].append(df_Sa0['monkey'][idx])
                
        df_serial = pd.DataFrame(serial)
        
        
        mse_acc_prev, mse_std_prev, mse_acc_curr, mse_std_curr = decode_continuous_prev_cur(df_serial)
        
        #print('save file')
        file1 = open("ContinuousDecoding_AllMonkeys.txt", "w")
        str_dictionary = repr(mse_acc_prev)
        file1.write(str_dictionary + "\n")
        str_dictionary = repr(mse_std_prev)
        file1.write(str_dictionary + "\n")
        str_dictionary = repr(mse_acc_curr)
        file1.write(str_dictionary + "\n")
        str_dictionary = repr(mse_std_curr)
        file1.write(str_dictionary + "\n")
        file1.close()
        
        print('saved monkey '+str(m)+str(n))