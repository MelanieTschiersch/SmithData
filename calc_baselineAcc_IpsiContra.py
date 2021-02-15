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
import h5py

##################################################################################################
#                                               FUNCTIONS                                        #
##################################################################################################

# y is fixed only X changes based on time in trial
def decoder_delayVSresponse(dataframe,borders_full):
    mode='MSE'

    y = dataframe['target_prev'].values-np.pi# target_prev # response_prev

    # train in previous trial delay (mean over individual delay length )
    X_delay = pd.DataFrame([np.mean(dataframe['bin_sp_prev'][n][borders_full[2]:borders_full[3]], axis=0) for n in range(len(dataframe['bin_sp_prev']))])
    # train in previous trial response (ITI=borders[6:7], reward= borders[5:6])
    X_response = pd.DataFrame([np.mean(dataframe['bin_sp_prev'][n][borders_full[6]:borders_full[7]],axis=0) for n in range(len(dataframe['bin_sp_prev']))])

    acc_bias_prev_delay = []
    std_bias_prev_delay = []
    acc_bias_curr_delay = []
    std_bias_curr_delay = []
    acc_bias_prev_response=[]
    std_bias_prev_response=[]
    acc_bias_curr_response=[]
    std_bias_curr_response=[]
    for delta_t_train in range(len(dataframe['bin_sp_prev'][dataframe['bin_sp_prev'].index[0]])):# for each period ['targ_on', 'targ_off', 'go_cue', 'saccade', 'reward']
        # create training dataset: columns=neurons, rows=trials within specified time bin(in period)
        X_prev = pd.DataFrame([dataframe['bin_sp_prev'][n][delta_t_train] for n in range(len(dataframe['bin_sp_prev']))])
        X_curr = pd.DataFrame([dataframe['bin_sp_curr'][n][delta_t_train] for n in range(len(dataframe['bin_sp_curr']))])
        # Crossvalidation
        acc_crosscorr_prev_delay=[]
        acc_crosscorr_curr_delay=[]
        acc_crosscorr_prev_response=[]
        acc_crosscorr_curr_response=[] 
        for k in range(0,1000):# k=10 as in Barbosa2020
            #train test split
            rand_state = randint(0,10000)# to get the same split for previous, current
            X_train_delay, X_i, y_train, y_i = train_test_split(X_delay, y, test_size = 0.20, random_state=rand_state)# train in slow loop
            X_train_response, X_i, y_train, y_i = train_test_split(X_response, y, test_size = 0.20, random_state=rand_state)# train in slow loop
            X_i, X_test_prev, y_i, y_test = train_test_split(X_prev, y, test_size = 0.20, random_state=rand_state)# test in fast loop
            X_i, X_test_curr, y_i, y_test = train_test_split(X_curr, y, test_size = 0.20, random_state=rand_state)# test in fast loop

            #y_test_cos = np.cos(y_test)
            
            # randomly shuffle train and test data
            X_train_delay = X_train_delay.sample(frac=1, axis=1).reset_index(drop=True)
            X_test_prev = X_test_prev.sample(frac=1, axis=1).reset_index(drop=True)
            X_train_response = X_train_response.sample(frac=1, axis=1).reset_index(drop=True)
            X_test_curr = X_test_curr.sample(frac=1, axis=1).reset_index(drop=True)
            np.random.shuffle(y_train)
            np.random.shuffle(y_test)
            
            # determine labels (sine, cosine)
            #assert (y_train_delay == y_train_response).all()
            y_train_cos = np.cos(y_train)
            y_train_sin = np.sin(y_train)

            # make linear regression fit for sine/cosine for prev/current trial
            model_d_cos = LinearRegression().fit(X_train_delay, y_train_cos)
            model_d_sin = LinearRegression().fit(X_train_delay, y_train_sin)
            model_r_cos = LinearRegression().fit(X_train_response, y_train_cos)
            model_r_sin = LinearRegression().fit(X_train_response, y_train_sin)

            # make predictions of models
            preds_prev_d_cos = model_d_cos.predict(X_test_prev)
            preds_prev_d_sin = model_d_sin.predict(X_test_prev)
            preds_curr_d_cos = model_d_cos.predict(X_test_curr)
            preds_curr_d_sin = model_d_sin.predict(X_test_curr)
            preds_prev_r_cos = model_r_cos.predict(X_test_prev)
            preds_prev_r_sin = model_r_sin.predict(X_test_prev)
            preds_curr_r_cos = model_r_cos.predict(X_test_curr)
            preds_curr_r_sin = model_r_sin.predict(X_test_curr)

            preds_d_prev = [math.atan2(preds_prev_d_sin[n],preds_prev_d_cos[n]) for n in range(len(preds_prev_d_sin))]
            preds_d_curr = [math.atan2(preds_curr_d_sin[n],preds_curr_d_cos[n]) for n in range(len(preds_curr_d_sin))]
            preds_r_prev = [math.atan2(preds_prev_r_sin[n],preds_prev_r_cos[n]) for n in range(len(preds_prev_r_sin))]
            preds_r_curr = [math.atan2(preds_curr_r_sin[n],preds_curr_r_cos[n]) for n in range(len(preds_curr_r_sin))]

            # R squared value
            if mode == 'r2':
                acc_crosscorr_prev_delay.append(metrics.r2_score(preds_d_prev, y_test.values))
                acc_crosscorr_curr_delay.append(metrics.r2_score(preds_d_curr, y_test.values))
                acc_crosscorr_prev_response.append(metrics.r2_score(preds_r_prev, y_test.values))
                acc_crosscorr_curr_response.append(metrics.r2_score(preds_r_curr, y_test.values))
            elif mode == 'MSE':
                #acc_crosscorr_prev_delay.append(np.mean((circdist(preds_d_prev, y_test.values)**2)))
                #acc_crosscorr_curr_delay.append(np.mean((circdist(preds_d_curr, y_test.values)**2)))
                #acc_crosscorr_prev_response.append(np.mean((circdist(preds_r_prev, y_test.values)**2)))
                #acc_crosscorr_curr_response.append(np.mean((circdist(preds_r_curr, y_test.values)**2)))
                acc_crosscorr_prev_delay.append(np.mean((circdist(preds_d_prev, y_test)**2)))
                acc_crosscorr_curr_delay.append(np.mean((circdist(preds_d_curr, y_test)**2)))
                acc_crosscorr_prev_response.append(np.mean((circdist(preds_r_prev, y_test)**2)))
                acc_crosscorr_curr_response.append(np.mean((circdist(preds_r_curr, y_test)**2)))
            else:
                print('Mode needs to be either:\n'+'r2 for evaluation with R² metric \n'+\
                          'or\nMSE for evaluation with mean-squared error.')
        acc_bias_prev_delay.append(np.mean(acc_crosscorr_prev_delay))
        std_bias_prev_delay.append(np.std(acc_crosscorr_prev_delay))
        acc_bias_curr_delay.append(np.mean(acc_crosscorr_curr_delay))
        std_bias_curr_delay.append(np.std(acc_crosscorr_curr_delay))

        acc_bias_prev_response.append(np.mean(acc_crosscorr_prev_response))
        std_bias_prev_response.append(np.std(acc_crosscorr_prev_response))
        acc_bias_curr_response.append(np.mean(acc_crosscorr_curr_response))
        std_bias_curr_response.append(np.std(acc_crosscorr_curr_response))
        
    return acc_bias_prev_delay,std_bias_prev_delay,acc_bias_curr_delay,std_bias_curr_delay,\
acc_bias_prev_response,std_bias_prev_response,acc_bias_curr_response,std_bias_curr_response


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
    

# CW vs CCW
x_start=[]
x_label=[]
clockw=[]

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
 
#only use Sa, sess0
df_Sa0_monkey = 'Sa'
df_Sa0_sess = 2
df_Sa0 = df_dat.loc[(df_dat['monkey']==df_Sa0_monkey) & (df_dat['session']==df_Sa0_sess)]

# make spike trains into csr matrix for each trial
mat = [csr_matrix(df_Sa0.loc[n,'sp_train']) for n in df_Sa0['sp_train'].index]
df_Sa0.loc[:,'n_mat'] = mat


# determine border points between different time periods, until beginning of delay
bins = 50 # TODO! 
timings = ['fix','targ_on','targ_off', 'go_cue']#,'saccade', 'reward', 'trial_end']# discrete timings
help = 0
borders=[]
# borders 
for period in range(len(timings)):
    #help += int(min([(df_Sa0.loc[n, timings[period+1]]-df_Sa0.loc[n, timings[period]]) for n in range(len(df_Sa0))])/bins)
    #borders.append(help)
    borders.append(int(min(df_dat[timings[period]]/bins)))
print(borders)

timings = ['fix','targ_on','targ_off', 'go_cue']#,'saccade', 'reward', 'trial_end']# discrete timings
help = 0
borders=[]
# borders 
for period in range(len(timings)):
    #help += int(min([(df_Sa0.loc[n, timings[period+1]]-df_Sa0.loc[n, timings[period]]) for n in range(len(df_Sa0))])/bins)
    #borders.append(help)
    borders.append(int(min(df_dat[timings[period]]/bins)))

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

## add shift between trial short end and trial long start
borders.append(borders[-1]+min(np.array(borders2['trial_end'])- np.array(borders2['delay_start'])))

# add saccade for response period
#borders.append(borders[-1]+min(np.array(borders2['saccade'])- np.array(borders2['delay_end'])))

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
    
# for first cut (different delay lengths)
bin_sp_trials_pastdelay=[]
period_spikes=[]
for idx, trial in enumerate(df_Sa0.index):# for all trials
    binned_spikes = []
    number_bins=[]
    for period in range(len(borders2)-1):# for all time periods until trial_end
        if period<4:
            number_bins.append(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0])
            for t in range(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0]): # for number of time bins in discrete timings:           
                # sum the matrix of neurons at timings in bin
                binned_spikes.append(np.sum(df_Sa0.loc[trial, 'n_mat'][:,borders2[t_borders2[period]][idx]*bins+t*bins:borders2[t_borders2[period]][idx]*bins+t*bins+bins].toarray(), axis=1))
        elif period>4:
            number_bins.append(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0])
            for t in range(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0]): # for number of time bins in discrete timings:           
                # sum the matrix of neurons at timings in bin
                binned_spikes.append(np.sum(df_Sa0.loc[trial, 'n_mat'][:,borders2[t_borders2[period]][idx]*bins+t*bins:borders2[t_borders2[period]][idx]*bins+t*bins+bins].toarray(), axis=1))

    #print(len(binned_spikes[0]))
    bin_sp_trials_pastdelay.append(binned_spikes)

bin_sp_complete = np.append(bin_sp_trials,bin_sp_trials_pastdelay, axis=1)

# add to dataframe
bin_s=[]
for trial,idx in enumerate(df_Sa0.index):
    bin_s.append(bin_sp_complete[trial])
df_Sa0['bin_sp']=bin_s

borders_full=[]
borders_full = np.append(borders[:-1],borders[-2]+number_bins[0])
for i in range(1,len(number_bins)):
    borders_full = np.append(borders_full,borders_full[-1]+number_bins[i])
#borders_full = np.append(borders_full,borders_full[-1]+borders2['reward'][0]-borders2['saccade'][0])
#borders_full = np.append(borders_full,borders_full[-1]+borders2['trial_end'][0]-borders2['reward'][0])
#borders_full = np.append(borders_full,borders_full[-1]+borders2['end'][0]-borders2['trial_end'][0])

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
        serial['targ_off_prev'].append(df_Sa0['targ_off'][idx])
        serial['go_cue_prev'].append(df_Sa0['go_cue'][idx])
        serial['response_prev'].append(df_Sa0['saccade_angle'][idx]*np.pi/180)
        serial['delay_prev'].append(df_Sa0['go_cue'][idx]-df_dat['targ_off'][idx])
        serial['bin_sp_prev'].append(bin_sp_complete[trial])
        serial['target_curr'].append(df_Sa0['targ_angle'][idx+1]*np.pi/180)
        serial['targ_on_curr'].append(df_Sa0['targ_on'][idx+1])
        serial['response_curr'].append(df_Sa0['saccade_angle'][idx+1]*np.pi/180)
        serial['delay_curr'].append(df_Sa0['go_cue'][idx+1]-df_Sa0['targ_off'][idx+1]) 
        serial['bin_sp_curr'].append(bin_sp_complete[trial+1])
        serial['monkey'].append(df_Sa0['monkey'][idx])
        serial['cw'].append((df_Sa0['clockw'][idx]=='CW'))
        
df_serial = pd.DataFrame(serial)


################# COMPARE HEMIFIELDS

# idx neurons each hemispheres
left = np.where(left_idx[df_Sa0_monkey][df_Sa0_sess]==1)[1]#
right = np.where(right_idx[df_Sa0_monkey][df_Sa0_sess]==1)[1]

# target position left/right
targ_left = np.where((df_serial['target_prev']>0) & (df_serial['target_prev']<np.pi))[0]
targ_right = np.where((df_serial['target_prev']>np.pi) & (df_serial['target_prev']<2*np.pi))[0] 

# create dataframe with only left neurons
df_serial_left = df_serial.copy()# ['bin_sp_prev'][0][0][left]
df_serial_left.drop(['bin_sp_prev'], axis=1)
df_serial_left['bin_sp_prev'] = [[df_serial['bin_sp_prev'][n][t][left] for t in range(len(df_serial['bin_sp_prev'][n]))] for n in range(len(df_serial['bin_sp_prev']))]
df_serial_left.drop(['bin_sp_curr'], axis=1)
df_serial_left['bin_sp_curr'] = [[df_serial['bin_sp_curr'][n][t][left] for t in range(len(df_serial['bin_sp_curr'][n]))] for n in range(len(df_serial['bin_sp_curr']))]

# only right neurons
df_serial_right = df_serial.copy()# ['bin_sp_prev'][0][0][left]
df_serial_right.drop(['bin_sp_prev'], axis=1)
df_serial_right['bin_sp_prev'] = [[df_serial['bin_sp_prev'][n][t][right] for t in range(len(df_serial['bin_sp_prev'][n]))] for n in range(len(df_serial['bin_sp_prev']))]
df_serial_right.drop(['bin_sp_curr'], axis=1)
df_serial_right['bin_sp_curr'] = [[df_serial['bin_sp_curr'][n][t][right] for t in range(len(df_serial['bin_sp_curr'][n]))] for n in range(len(df_serial['bin_sp_curr']))]



        

# left ipsi
df_serial_left_ipsi = df_serial_left.loc[targ_left].reset_index()#
label_ipsi = 'ipsi'

acc_bias_prev_delay_ipsi,std_bias_prev_delay_ipsi,acc_bias_curr_delay_ipsi,std_bias_curr_delay_ipsi,\
acc_bias_prev_response_ipsi,std_bias_prev_response_ipsi,acc_bias_curr_response_ipsi,std_bias_curr_response_ipsi = decoder_delayVSresponse(df_serial_left_ipsi, borders_full)

# left contra
df_serial_left_contra = df_serial_left.loc[targ_right].reset_index()

acc_bias_prev_delay_contra,std_bias_prev_delay_contra,acc_bias_curr_delay_contra,std_bias_curr_delay_contra,\
acc_bias_prev_response_contra,std_bias_prev_response_contra,acc_bias_curr_response_contra,std_bias_curr_response_contra = decoder_delayVSresponse(df_serial_left_contra, borders_full)
label_contra='contra'


hf = h5py.File('baseline_IpsiContra_Sa2_0top.h5', 'a')
hf.create_dataset('acc_prev_ipsiDelay', data=acc_bias_prev_delay_ipsi)
hf.create_dataset('std_prev_ipsiDelay', data=std_bias_prev_delay_ipsi)
hf.create_dataset('acc_curr_ipsiDelay', data=acc_bias_curr_delay_ipsi)
hf.create_dataset('std_curr_ipsiDelay', data=std_bias_curr_delay_ipsi)

hf.create_dataset('acc_prev_ipsiResponse', data=acc_bias_prev_response_ipsi)
hf.create_dataset('std_prev_ipsiResponse', data=std_bias_prev_response_ipsi)
hf.create_dataset('acc_curr_ipsiResponse', data=acc_bias_curr_response_ipsi)
hf.create_dataset('std_curr_ipsiResponse', data=std_bias_curr_response_ipsi)
#contra
hf.create_dataset('acc_prev_contraDelay', data=acc_bias_prev_delay_contra)
hf.create_dataset('std_prev_contraDelay', data=std_bias_prev_delay_contra)
hf.create_dataset('acc_curr_contraDelay', data=acc_bias_curr_delay_contra)
hf.create_dataset('std_curr_contraDelay', data=std_bias_curr_delay_contra)

hf.create_dataset('acc_prev_contraResponse', data=acc_bias_prev_response_contra)
hf.create_dataset('std_prev_contraResponse', data=std_bias_prev_response_contra)
hf.create_dataset('acc_curr_contraResponse', data=acc_bias_curr_response_contra)
hf.create_dataset('std_curr_contraResponse', data=std_bias_curr_response_contra)
hf.close()