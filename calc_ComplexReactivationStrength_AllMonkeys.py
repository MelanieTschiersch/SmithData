#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:23:15 2020

@author: melanie

calculate the prediction error (target) from a delay-decoder in a leave-1-out
cross validation during the reactivation period (borders_full[7]:borders_full[8]
of previous trial, borders_full[0]:borders_full[1] of current trial for all monkeys
"""

import numpy as np
import pandas as pd
from scipy.stats import *
from scipy.sparse import csr_matrix
from sklearn.model_selection import LeaveOneOut
from circ_stats import *
import h5py
import pickle
from sklearn import preprocessing
from numpy.linalg import inv

##################################################################################################
#                                               FUNCTIONS                                        #
##################################################################################################


# y is fixed only X changes based on time in trial
def decode_complex_reactivations(dataframe,borders_full,mode):
    '''Only decode during reactivation to check strength of serial dependence for trials
    INPUT:      dataframe = serial dependence dataframe
                y = target/response position per trial in rad [-np.pi,np.pi]
                borders_full = borders of time periods 
                mode = MSE, r² method of evaluation
    OUTPUT:     acc_curr = array of circular distance between prediction, target for time x trial
    '''
    # y is fixed only X changes based on time in trial
    y = np.array([complex(dataframe['target_prev_xy'][i][0],dataframe['target_prev_xy'][i][1]) for i in dataframe.index])

    acc = []
    acc_dist=[]
    
    # use delay decoder
    X_delay = pd.DataFrame([np.mean(dataframe['bin_sp_prev'][n][borders_full[3]:borders_full[4]], axis=0) for n in range(len(dataframe['bin_sp_prev']))])
    #scaler = preprocessing.MinMaxScaler()
    #d = scaler.fit_transform(X_delay)
    #X_delay = pd.DataFrame(d, columns=X_delay.columns)


    # timing in 250ms before current stimulus start until current target start
    timing = np.append(np.arange(borders_full[8], borders_full[9]), np.arange(0,borders_full[3]))
    for delta_t_train in timing:# for trial start to target on, current trial
        # create training dataset: columns=neurons, rows=trials for previous/current trials
        if delta_t_train >= borders_full[8]:
            X_prev = pd.DataFrame([dataframe['bin_sp_prev'][n][delta_t_train] for n in dataframe['bin_sp_prev'].index])
            #d = scaler.fit_transform(X_prev)
            #X_prev = pd.DataFrame(d, columns=X_prev.columns)
            
            # Crossvalidation
            acc_crosscorr=[]
            acc_crosscorr_dist=[]
            
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(X_prev):
                X_train_delay, X_test_prev = X_delay.loc[train_idx], X_prev.loc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # train model
                weights_prev = np.linalg.pinv(X_train_delay.T.dot(X_train_delay)).dot(X_train_delay.T).dot(y_train)     
                #print(weights_prev[0])
                # test correctness of predictions
                if mode == 'r2':
                    acc_crosscorr.append(metrics.r2_score(preds_curr, y_test))
                elif mode == 'MSE':
                    acc_crosscorr.append(circdist(np.angle(weights_prev.dot(X_test_prev.T)),np.angle(y_test)))
                    acc_crosscorr_dist.append(abs(weights_prev.dot(X_test_prev.T))-abs(y_test))                    
                else:
                    print('Mode needs to be either:\n'+'r2 for evaluation with R² metric \n'+\
                                  'or\nMSE for evaluation with mean-squared error.')
                    return
                
        else:
            X_curr = pd.DataFrame([dataframe['bin_sp_curr'][n][delta_t_train] for n in dataframe['bin_sp_curr'].index])
            #d = scaler.fit_transform(X_curr)
            #X_curr = pd.DataFrame(d, columns=X_curr.columns)
            
            # Crossvalidation
            acc_crosscorr=[]
            acc_crosscorr_dist=[]
        
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(X_prev): # for start of trial only decode during current trial for reactivations
                X_train_curr, X_test_curr = X_delay.loc[train_idx], X_curr.loc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                weights_curr = np.linalg.pinv(X_train_curr.T.dot(X_train_curr)).dot(X_train_curr.T).dot(y_train)     
                #print(weights_curr)
                
                # test correctness of predictions
                if mode == 'r2':
                    acc_crosscorr.append(metrics.r2_score(preds_curr, y_test))
                elif mode == 'MSE':
                    acc_crosscorr.append(circdist(np.angle(weights_curr.dot(X_test_curr.T)),np.angle(y_test)))
                    acc_crosscorr_dist.append(abs(weights_curr.dot(X_test_curr.T))-abs(y_test))                    
                else:
                    print('Mode needs to be either:\n'+'r2 for evaluation with R² metric \n'+\
                                  'or\nMSE for evaluation with mean-squared error.')
                    return
                
        acc.append(acc_crosscorr)
        acc_dist.append(acc_crosscorr_dist)
        #std_curr.append(np.std(acc_crosscorr_curr))
    return acc, acc_dist



##################################################################################################
#                                               LOAD DATA                                        #
##################################################################################################


with open('../../Data/df.pickle', 'rb') as handle:
    df_dat = pickle.load(handle)
    
with open('../../Data/leftRightIdx.pickle', 'rb') as handle:
    leftRightIdx = pickle.load(handle)
    
left_idx = {'Sa': [[] for i in range(len(leftRightIdx['left']['Sa']))], 'Pe':[[] for i in range(len(leftRightIdx['left']['Pe']))], 'Wa':[[] for i in range(len(leftRightIdx['left']['Wa']))]}
right_idx = {'Sa': [[] for i in range(len(leftRightIdx['left']['Sa']))], 'Pe':[[] for i in range(len(leftRightIdx['left']['Pe']))], 'Wa':[[] for i in range(len(leftRightIdx['left']['Wa']))]}
for m in ["Sa", "Pe", "Wa"]:
    for n in range(len(leftRightIdx['left'][m])):
        left_idx[m][n] = leftRightIdx['left'][m][n]
        right_idx[m][n] = leftRightIdx['right'][m][n]

df_dat['broke'] = [True if df_dat['outcome'][n]!='CORRECT' else False for n in df_dat.index]


##################################################################################################
#                                               SPLIT DATA                                        #
##################################################################################################
 
for mono in ['Sa']:#, 'Pe', 'Wa'
    for sess in range(0,max(df_dat['session'].loc[df_dat['monkey']==mono])+1): 
        print(sess)
        #only use Sa, sess0
        #df_Sa0 = df_dat.loc[(df_dat['monkey']==mono) & (df_dat['session']==sess)]
        
        df_Sa0_corr = df_dat.loc[(df_dat['monkey']==mono) & (df_dat['session']==sess) & (df_dat['outcome']=='CORRECT')]
        
        # make spike trains into csr matrix for each trial
        mat = [csr_matrix(df_Sa0_corr.loc[n,'sp_train']) for n in df_Sa0_corr['sp_train'].index]
        df_Sa0_corr['n_mat'] = mat
        
        
        # determine border points between different time periods, until beginning of delay
        bins = 50 # TODO! 
        timings = ['start','fix','targ_on','targ_off', 'go_cue']#,'saccade', 'reward', 'trial_end']# discrete timings
        help = 0
        borders=[]
        # borders 
        for period in range(len(timings)):
            #help += int(min([(df_Sa0.loc[n, timings[period+1]]-df_Sa0.loc[n, timings[period]]) for n in range(len(df_Sa0))])/bins)
            #borders.append(help)
            borders.append(int(min(df_Sa0_corr[timings[period]]/bins)))
        
        # determine border points INDIVID trials between different time periods, for end of delay
        timings2 = ['go_cue','saccade', 'reward', 'trial_end']
        t_borders2 = ['delay_start','delay_end','saccade', 'reward', 'trial_end', 'end_start', 'end']#
        borders2={'delay_start': [], 'delay_end': [], 'saccade': [], 'reward':[], 'trial_end':[], 'end_start':[], 'end':[]}##np.zeros((len(timings2)+1, len(df_Sa0)))
        for i,m in enumerate(borders2.keys()):
            if i==0:
                #create shifted "start" of delay
                borders2[m] = ((df_Sa0_corr['go_cue'].values)/bins - min(((df_Sa0_corr['go_cue'].values-df_Sa0_corr['targ_off'].values)/bins))).astype(int)#
            elif i ==1:
                # delay end
                borders2[m] = ((df_Sa0_corr['go_cue'].values)/bins).astype(int)
                #np.array([int(df_Sa0.loc[n,timings2[0]]/bins)-(borders[-1]) for n in range(len(df_Sa0))])
            elif m =='end_start':
                # shifted "start" of trial end : complete end of trial - minimum(trial_end-reward)
                borders2[m] = [int(df_Sa0_corr.loc[n,'trial_end']/bins)-int(min((df_Sa0_corr.loc[:,'trial_end']-df_Sa0_corr.loc[:,'reward'])/bins)) for n in df_Sa0_corr.index]#
            elif m == 'end':
                borders2[m] = [int(df_Sa0_corr.loc[n,'trial_end']/bins) for n in df_Sa0_corr.index]
            else:
                # create end delay, saccade start, reward start, trial_end through using minimum distance between periods, adding to delay_end, saccade_end,..
                borders2[m] = np.array(borders2[t_borders2[i-1]]) + min([int((df_Sa0_corr.loc[n,timings2[i-1]]-df_Sa0_corr.loc[n,timings2[i-2]])/bins) for n in df_Sa0_corr.index])
        
        ## add shift between trial short end and trial long start
        borders.append(borders[-1]+min(np.array(borders2['trial_end'])- np.array(borders2['delay_start'])))
        print(borders)
        
        # add saccade for response period
        #borders.append(borders[-1]+min(np.array(borders2['saccade'])- np.array(borders2['delay_end'])))
        
        bin_sp_trials=[]
        period_spikes=[]
        for trial in df_Sa0_corr.index:# for all trials
            binned_spikes = []
            for period in range(len(timings[:-1])):# for all discrete timings
                for t in range(borders[period+1]-borders[period]): # for all time bins in discrete timings:           
                    # sum the matrix of neurons at timings in bin
                    binned_spikes.append(np.sum(df_Sa0_corr.loc[trial, 'n_mat'][:,int(df_Sa0_corr.loc[trial,timings[period]]+t*bins):int(df_Sa0_corr.loc[trial,timings[period]]+t*bins+bins)].toarray(), axis=1))
                #print(t)
            #print(len(binned_spikes[0]))
            bin_sp_trials.append(binned_spikes)
            
        # for first cut (different delay lengths)
        bin_sp_trials_pastdelay=[]
        period_spikes=[]
        for idx, trial in enumerate(df_Sa0_corr.index):# for all trials
            binned_spikes = []
            number_bins=[]
            for period in range(len(borders2)-1):# for all time periods until trial_end
                if period<4:
                    number_bins.append(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0])
                    for t in range(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0]): # for number of time bins in discrete timings:           
                        # sum the matrix of neurons at timings in bin
                        binned_spikes.append(np.sum(df_Sa0_corr.loc[trial, 'n_mat'][:,borders2[t_borders2[period]][idx]*bins+t*bins:borders2[t_borders2[period]][idx]*bins+t*bins+bins].toarray(), axis=1))
                elif period>4:
                    number_bins.append(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0])
                    for t in range(borders2[t_borders2[period+1]][0]-borders2[t_borders2[period]][0]): # for number of time bins in discrete timings:           
                        # sum the matrix of neurons at timings in bin
                        binned_spikes.append(np.sum(df_Sa0_corr.loc[trial, 'n_mat'][:,borders2[t_borders2[period]][idx]*bins+t*bins:borders2[t_borders2[period]][idx]*bins+t*bins+bins].toarray(), axis=1))
        
            #print(len(binned_spikes[0]))
            bin_sp_trials_pastdelay.append(binned_spikes)
        
        bin_sp_complete = np.append(bin_sp_trials,bin_sp_trials_pastdelay, axis=1)
        
        # add to dataframe
        bin_s=[]
        for trial,idx in enumerate(df_Sa0_corr.index):
            bin_s.append(bin_sp_complete[trial])
        df_Sa0_corr['bin_sp']=bin_s
        
        borders_full=[]
        borders_full = np.append(borders[:-1],borders[-2]+number_bins[0])
        for i in range(1,len(number_bins)):
            borders_full = np.append(borders_full,borders_full[-1]+number_bins[i])
        
        borders_pastdelay = borders_full[len(borders):]

        
        ##################################################################################################
        #                                          SERIAL BIAS                                           #
        ##################################################################################################
        
        serial = {'trial_id':[], 'outcome':[], 'target_prev': [],'target_prev_xy': [],  'targ_off_prev':[], 'go_cue_prev':[], 'response_prev': [],\
                  'delay_prev': [],'bin_sp_prev':[], 'target_curr': [], 'target_curr_xy': [], 'targ_on_curr':[], 'response_curr': [],\
                  'delay_curr': [], 'bin_sp_curr':[], 'ITI': [], 'broke': [], 'monkey': []}
        
        #for trial,idx in enumerate(df_Sa0.index[:-1]):
        #    if ((df_Sa0['trial_id'][idx]+1) == (df_Sa0['trial_id'][idx+1])):
        df_Sa0_corr_reset = df_Sa0_corr.copy().reset_index()
        
        cut_off_time=5
        for idx in df_Sa0_corr_reset.index[:-1]:# run through all correct trials (0,len)
            if df_Sa0_corr_reset.loc[idx,'trial_id'] < df_Sa0_corr_reset.loc[idx+1,'trial_id']: # only compare within one sesssion
                if np.sum(df_dat[df_Sa0_corr_reset.loc[idx,'index']+1:df_Sa0_corr_reset.loc[idx+1,'index']]['trial_end'])<cut_off_time: # only use trials with less than cut_off ms between 2 correct trials
                    serial['trial_id'].append(idx)
                    serial['outcome'].append(df_Sa0_corr_reset['outcome'][idx])
                    serial['target_prev'].append(df_Sa0_corr_reset['targ_angle'][idx]*np.pi/180)
                    serial['target_prev_xy'].append(df_Sa0_corr_reset['targ_xy'][idx])
                    serial['targ_off_prev'].append(df_Sa0_corr_reset['targ_off'][idx])
                    serial['go_cue_prev'].append(df_Sa0_corr_reset['go_cue'][idx])
                    serial['response_prev'].append(df_Sa0_corr_reset['saccade_angle'][idx]*np.pi/180)
                    serial['delay_prev'].append(df_Sa0_corr_reset['go_cue'][idx]-df_dat['targ_off'][idx])
                    serial['bin_sp_prev'].append(bin_sp_complete[idx])
                    serial['target_curr'].append(df_Sa0_corr_reset['targ_angle'][idx+1]*np.pi/180)
                    serial['target_curr_xy'].append(df_Sa0_corr_reset['targ_xy'][idx+1])
                    serial['targ_on_curr'].append(df_Sa0_corr_reset['targ_on'][idx+1])
                    serial['response_curr'].append(df_Sa0_corr_reset['saccade_angle'][idx+1]*np.pi/180)
                    serial['delay_curr'].append(df_Sa0_corr_reset['go_cue'][idx+1]-df_Sa0_corr_reset['targ_off'][idx+1]) 
                    serial['bin_sp_curr'].append(bin_sp_complete[idx+1])
                    serial['ITI'].append((df_Sa0_corr_reset['trial_end'][idx]-df_Sa0_corr_reset['go_cue'][idx])+np.sum(df_Sa0_corr[df_Sa0_corr_reset.loc[idx,'index']+1:df_Sa0_corr_reset.loc[idx+1,'index']]['reward']) + (df_Sa0_corr_reset['targ_on'][idx+1]-df_Sa0_corr_reset['start'][idx+1]))# ITI time is time after reward + broken off fixations
                    serial['broke'].append(df_Sa0_corr_reset.loc[idx+1,'index']- (df_Sa0_corr_reset.loc[idx,'index']+1))# how many broken trials btwn 2 correct trials
                    serial['monkey'].append(df_Sa0_corr_reset['monkey'][idx])
                
        df_serial = pd.DataFrame(serial)

        ##################################################################################################
        #                                          SERIAL BIAS DEPENDENCY                                  #
        ##################################################################################################      
        
        mode='MSE'
        acc_reactivations_singletrial,\
        acc_reactivations_singletrial_dist = decode_complex_reactivations(df_serial,borders_full,mode)
        
        acc_reactivations_singletrial = np.array(acc_reactivations_singletrial)
        acc_reactivations_singletrial = np.concatenate(acc_reactivations_singletrial, axis=1)

        acc_reactivations_singletrial_dist = np.array(acc_reactivations_singletrial_dist)
        acc_reactivations_singletrial_dist = np.concatenate(acc_reactivations_singletrial_dist, axis=1)

        ##################################################################################################
        #                                          SAVE DATA                                             #
        ##################################################################################################
        #file1 = open("reactivationStrength_AllMonkeys.txt", "a+")
        ## left ipsi
        #str_dictionary = repr(acc_reactivations_singletrial)
        #file1.write(str_dictionary + "\n")
        hf = h5py.File('ComplexReactivationStrength_AllMonkeys.h5', 'a')
        hf.create_dataset(mono+str(sess), data=acc_reactivations_singletrial)
        hf.create_dataset(mono+str(sess)+'_dist', data=acc_reactivations_singletrial_dist)
        hf.create_dataset(mono+str(sess)+'_borders', data=borders_full)
        hf.close()
        
        print('saved monkey '+str(mono)+str(sess))