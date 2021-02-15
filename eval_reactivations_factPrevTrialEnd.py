#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:32:32 2021

@author: melanie
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import *
from scipy.optimize import curve_fit
from cmath import phase
from numpy import array
from scipy.sparse import csr_matrix
import urllib
import glob
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn import metrics
from random import randint
import math
import io
import h5py
from circ_stats import *


def read_reactivations_AllMonkeys(filename, df):
    ''' 
    reads in prediction error of target, prediction from leave-1-out crossval from delay-decoder
    during the reactivation period (5 sessions)
    INPUT:
        filename:       name of file containing decoding accuracies .h5 file
                        (bins=50ms)  
        df:             dataframe containing minimum monkey, session information
    OUTPUT:
        acc:            prediction error [monkey,session x trials x reactivation time]
    
    '''
    hf = h5py.File(filename, 'r')

    acc = {'Sa':[], 'Pe':[], 'Wa':[]}
    borders = {'Sa':[], 'Pe':[], 'Wa':[]}
    print(hf.keys())
    for mono in ['Sa', 'Pe', 'Wa']:
        for sess in range(max(df.loc[df.monkey==mono].session)+1):
            n1 = hf.get(mono+str(sess))
            print(n1)
            acc[mono].append(np.array(n1))
            
            # get borders of time period endings
            n2 = hf.get(mono+str(sess)+'_borders')
            #print(n1)
            borders[mono].append(np.array(n2))

    hf.close()
    
    return acc, borders


def calc_errorcurve(results):
    '''
    Calculates single-sided serial dependence curve

    Parameters
    ----------
    results :   monkey data for [subject][params], with params:
                            target1:   target shown in previous trial
                            target2:   target shown in current trial
                            response1: response given in previous trial
                            response2: response given in current trial
                results : {"a": {"target1": [],"target2": [],"response1": [],"response2": []}, 
                           "l": {"target1": [],"target2": [],"response1": [],"response2": []}}
                w1      : smoothing window width
                w2      : smoothing window shift

    Returns
    -------
    err_avg : average smoothed error per monkey 
    rel_loc_avg : relative location equally spaced in [0,pi]
    err_std : sem of smoothed error
    err : original (unsmoothed) flipped error to (0,pi)

    '''
    rel_loc=[]
    err = []
    rel_loc.append(circdist(results['target_curr'].values,results['target_prev'].values))# relative location current prvious stimulus
    err.append(circdist(results['target_curr'].values,results['response_curr'].values))# error current trial
    # create half curve
    err = np.squeeze(err)#np.squeeze(np.sign(rel_loc)*err)# flip error along x-axis
    rel_loc = np.squeeze(np.round(rel_loc,3))#np.squeeze(np.sign(rel_loc)*rel_loc)# flip location along y-axis
    return rel_loc, err


############################# LOAD DATA ##################################
    
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

# CHANGE WM circle so that 0 top

help_circle = np.zeros(len(df_dat))
help_circle_sac = np.zeros(len(df_dat))
for idx in range(len(df_dat)):
    if df_dat.targ_angle[idx]<90:
        help_circle[idx] = df_dat.targ_angle[idx]+270
        help_circle_sac[idx] = df_dat.saccade_angle[idx]+270
    elif df_dat.targ_angle[idx]==90:
        help_circle[idx] = 0
        help_circle_sac[idx] = 0
    else:
        help_circle[idx] = df_dat.targ_angle[idx]-90
        help_circle_sac[idx] = df_dat.saccade_angle[idx]-90

df_dat['target_0top'] = help_circle
df_dat['saccade_0top'] = help_circle_sac
    

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



# create list with values needed for serial bias calculation (ONLY if two trials are in a row)
# no need to extra specify sessions, trial_id changes from large # to small #
serial_all = {'trial_id':[], 'target_prev': [], 'response_prev': [], 'delay_prev': [], 'target_curr': [], 'response_curr': [], 'delay_curr': [], 'monkey': [], 'session':[]}
for idx in df_dat.index[:-1]:
    if ((df_dat['trial_id'][idx]+1) == (df_dat['trial_id'][idx+1])):
        serial_all['trial_id'].append(idx)
        serial_all['target_prev'].append(df_dat['target_0top'][idx]*np.pi/180)
        serial_all['response_prev'].append(df_dat['saccade_0top'][idx]*np.pi/180)
        serial_all['delay_prev'].append(df_dat['go_cue'][idx]-df_dat['targ_off'][idx])
        serial_all['target_curr'].append(df_dat['target_0top'][idx+1]*np.pi/180)
        serial_all['response_curr'].append(df_dat['saccade_0top'][idx+1]*np.pi/180)
        serial_all['delay_curr'].append(df_dat['go_cue'][idx+1]-df_dat['targ_off'][idx+1]) 
        serial_all['monkey'].append(df_dat['monkey'][idx])
        serial_all['session'].append(df_dat['session'][idx])

df_serial_all = pd.DataFrame(serial_all)

# Compute serial dependence
rel_loc_all, err_all = calc_errorcurve(df_serial_all)
        
sb_all = {'rel_loc': rel_loc_all*180/np.pi, 'err': err_all*180/np.pi, 'delay_prev': serial_all['delay_prev'],\
      'delay_curr': serial_all['delay_curr'], 'monkey': serial_all['monkey']}
df_sb_all = pd.DataFrame(sb_all)

sb_onesided_all = {'rel_loc': rel_loc_all*180/np.pi*np.sign(rel_loc_all),\
               'err': np.array([err_all[i]*180/np.pi*np.sign(rel_loc_all[i]) if rel_loc_all[i] !=0 else err_all[i]*180/np.pi for i in range(len(rel_loc_all))]),\
               'delay_prev': serial_all['delay_prev'], 'delay_curr': serial_all['delay_curr'], 'monkey': serial_all['monkey']}
df_sb_onesided_all = pd.DataFrame(sb_onesided_all)


##################### START FACT ##################

# accuracy (MSE) sorted by monkey
acc_react_all, borders_all = read_reactivations_AllMonkeys('../Results/reactivationStrength2cue_AllMonkeys.h5', df_dat)#reactivationStrengthAnd2cue_AllMonkeys.h5
#back_until = 16#*200ms, how far into past (before 2nd trial start) the minimum is computed

bins=50#ms

ttest=[]
pval=[]
steps=17#48 for reactivationStrengthAnd2cue_AllMonkeys# 17 for reactivationStrength2cue_AllMonkeys
#for fact in range(0,100,steps):# how much percent of trial_end time of the previous trial is used in determination of minimum
da=[]
num_high180=[]
num_high135=[]
num_high90=[]
num_high45=[]
num_high0=[]
num_low180=[]
num_low135=[]
num_low90=[]
num_low45=[]
num_low0=[]
for back_until in range(1,steps):
    high_decoding_idx_full=[]#{'Sa':[],'Pe':[],'Wa':[]}
    low_decoding_idx_full=[]#{'Sa':[],'Pe':[],'Wa':[]}
    high_trials ={'Sa':[],'Pe':[],'Wa':[]}
    low_trials={'Sa':[],'Pe':[],'Wa':[]}
    high_trials_conc =[]
    low_trials_conc=[]
    len_sess=[]
    #print(fact)
    for mono in ['Sa', 'Pe', 'Wa']:
        for sess in range(max(df_dat.loc[df_dat.monkey==mono].session)+1):
            #back_until = int(fact/100*(borders_all[mono][sess][8]-borders_all[mono][sess][7]))# go back half the trial_end time
            
            #print(np.min(acc_react_all[mono][sess][:, (borders_all[mono][sess][8]-borders_all[mono][sess][7]-back_until):], axis=1))
            if back_until==1:
                acc_react_one_avg = np.min(abs(acc_react_all[mono][sess][:, -(back_until):]), axis=1)
            else:
                acc_react_one_avg = np.min(abs(acc_react_all[mono][sess][:, -(back_until):-(back_until-1)]), axis=1)#
            da.append((np.array(abs(acc_react_all[mono][sess][:, -(back_until):-(back_until-1)])).shape))
            # split into high VS low decoding
            cut = np.median(acc_react_one_avg)
            #print(cut)
            high_decoding_idx = np.where(acc_react_one_avg<cut)[0]
            low_decoding_idx = np.where(acc_react_one_avg>=cut)[0]
            #print(high_decoding_idx+sum(len_sess))
            #print(low_decoding_idx+sum(len_sess))
            #high_decoding_idx_full[mono].append(high_decoding_idx)
            #low_decoding_idx_full[mono].append(low_decoding_idx)
            high_decoding_idx_full.append(high_decoding_idx+sum(len_sess))
            low_decoding_idx_full.append(low_decoding_idx+sum(len_sess))
            len_sess.append(len(acc_react_all[mono][sess]))
    
            high_trials[mono].append(acc_react_all[mono][sess][high_decoding_idx,:])
            low_trials[mono].append(acc_react_all[mono][sess][low_decoding_idx,:])
            high_trials_conc.append(acc_react_all[mono][sess][high_decoding_idx,:])
            low_trials_conc.append(acc_react_all[mono][sess][low_decoding_idx,:])
            
    #high decoding
    # TODO! CHANGE HOW MANY MONKEYS ARE INCLUDED IN ANALYSIS
    high = np.concatenate(high_decoding_idx_full[0:3]).astype(int)# 0:3 = Sa
    df_high_all = df_serial_all.loc[high]#.reset_index()
    #rel_loc_high_all, err_high_all = calc_errorcurve(df_high_all)
   
    sb_high_all = {'rel_loc': rel_loc_all[high]*180/np.pi, 'err': err_all[high]*180/np.pi, 'delay_prev': df_high_all['delay_prev'],\
          'delay_curr': df_high_all['delay_curr'], 'monkey':df_high_all['monkey']}
    df_sb_high_all = pd.DataFrame(sb_high_all)
    
    sb_onesided_high_all = {'rel_loc': rel_loc_all[high]*180/np.pi*np.sign(rel_loc_all[high]),\
                   'err': np.array([err_all[high][i]*180/np.pi*np.sign(rel_loc_all[high][i]) if rel_loc_all[high][i] !=0 else err_all[high][i]*180/np.pi for i in range(len(rel_loc_all[high]))]),\
                   'delay_prev': df_high_all['delay_prev'], 'delay_curr': df_high_all['delay_curr'],\
                   'monkey': df_high_all['monkey']}
    df_sb_onesided_high_all = pd.DataFrame(sb_onesided_high_all)
    
    
    # low decoding
    # TODO! CHANGE HOW MANY MONKEYS ARE INCLUDED IN ANALYSIS
    low = np.concatenate(low_decoding_idx_full[0:3]).astype(int)
    df_low_all = df_serial_all.loc[low]#.reset_index()
    rel_loc_low_all, err_low_all = calc_errorcurve(df_low_all)
    
    sb_low_all = {'rel_loc': rel_loc_all[low]*180/np.pi, 'err': err_all[low]*180/np.pi, 'delay_prev': df_low_all['delay_prev'],\
          'delay_curr': df_low_all['delay_curr'], 'monkey':df_low_all['monkey']}
    df_sb_low_all = pd.DataFrame(sb_low_all)
    
    sb_onesided_low_all = {'rel_loc': rel_loc_all[low]*180/np.pi*np.sign(rel_loc_all[low]),\
                   'err': np.array([err_all[low][i]*180/np.pi*np.sign(rel_loc_all[low][i]) if rel_loc_all[low][i] !=0 else err_all[low][i]*180/np.pi for i in range(len(rel_loc_all[low]))]),\
                   'delay_prev': df_low_all['delay_prev'], 'delay_curr': df_low_all['delay_curr'],\
                   'monkey': df_low_all['monkey']}
    df_sb_onesided_low_all = pd.DataFrame(sb_onesided_low_all)
    
    
    num_high180.append(len(np.where(np.round(sb_onesided_high_all['rel_loc'])==180)[0]))
    num_high135.append(len(np.where(np.round(sb_onesided_high_all['rel_loc'])==135)[0]))
    num_high90.append(len(np.where(np.round(sb_onesided_high_all['rel_loc'])==90)[0]))
    num_high45.append(len(np.where(np.round(sb_onesided_high_all['rel_loc'])==45)[0]))
    num_high0.append(len(np.where(np.round(sb_onesided_high_all['rel_loc'])==0)[0]))
    
    num_low180.append(len(np.where(np.round(sb_onesided_low_all['rel_loc'])==180)[0]))
    num_low135.append(len(np.where(np.round(sb_onesided_low_all['rel_loc'])==135)[0]))
    num_low90.append(len(np.where(np.round(sb_onesided_low_all['rel_loc'])==90)[0]))
    num_low45.append(len(np.where(np.round(sb_onesided_low_all['rel_loc'])==45)[0]))
    num_low0.append(len(np.where(np.round(sb_onesided_low_all['rel_loc'])==0)[0]))
    
    ### Define curves to fit
    # use DoG
    def test_func(x,a,w):
        return a * w * x * (np.sqrt(2)/np.exp(-0.5)) * np.exp(-(w*x)**2)
    
    
    # fit curve
    params_high_all, param_cov_high_all = curve_fit(test_func, sb_high_all['rel_loc'], sb_high_all['err'], p0=[5, 0.015])#p0 = [amplitude, width] 
    params_onesided_high_all, param_onesided_cov_high_all = curve_fit(test_func, sb_onesided_high_all['rel_loc'], sb_onesided_high_all['err'], p0=[5, 0.015])
    
    params_low_all, param_cov_low_all = curve_fit(test_func, sb_low_all['rel_loc'], sb_low_all['err'], p0=[5, 0.015])#p0 = [amplitude, width] 
    params_onesided_low_all, param_onesided_cov_low_all = curve_fit(test_func, sb_onesided_low_all['rel_loc'], sb_onesided_low_all['err'], p0=[5, 0.015])
    
    ##########################################################################
    #                              PLOTS                                     #
    ##########################################################################

    # plot serial bias curve, both sides
    x_new = np.linspace(-180, 180, 1000)
    x_new_onesided = np.linspace(0, 180, 500)
    num_pos = len(df_sb_high_all.groupby('rel_loc')) # number of x-positions in SB curve
    x_full = np.linspace(-180,180,num_pos)
    x_full_onesided = np.linspace(0,180,int(num_pos/2+1))
    
    # plt.figure(figsize=(15,5))
    # # plot serial bias curve, single sided
    # plt.axhline(color='grey', linewidth=1)
    # plt.axvline(color='grey', linewidth=1)
    # plt.scatter(sb_onesided_high_all['rel_loc'], sb_onesided_high_all['err'], color='lightcoral', alpha=0.2)
    # plt.fill_between(x_full_onesided,df_sb_onesided_high_all.groupby('rel_loc').median()['err']+df_sb_onesided_high_all.groupby('rel_loc').sem()['err'], df_sb_onesided_high_all.groupby('rel_loc').median()['err']-df_sb_onesided_high_all.groupby('rel_loc').sem()['err'], color='lightcoral', alpha=0.3)
    # plt.plot(x_full_onesided,df_sb_onesided_high_all.groupby('rel_loc').median()['err'], color='lightcoral', label='median,high')
    # # DoG fit of data
    # plt.plot(x_new_onesided, test_func(x_new_onesided, params_onesided_high_all[0], params_onesided_high_all[1]), label='DoG,high',color='darkred')
    
    # plt.scatter(sb_onesided_low_all['rel_loc'], sb_onesided_low_all['err'], color='lightblue', alpha=0.2)
    # plt.fill_between(x_full_onesided,df_sb_onesided_low_all.groupby('rel_loc').median()['err']+df_sb_onesided_low_all.groupby('rel_loc').sem()['err'], df_sb_onesided_low_all.groupby('rel_loc').median()['err']-df_sb_onesided_low_all.groupby('rel_loc').sem()['err'], color='lightblue', alpha=0.3)
    # plt.plot(x_full_onesided,df_sb_onesided_low_all.groupby('rel_loc').median()['err'], color='lightblue', label='median,low')
    # # DoG fit of data
    # plt.plot(x_new_onesided, test_func(x_new_onesided, params_onesided_low_all[0], params_onesided_low_all[1]), label='DoG,low',color='darkblue')
    # plt.xlabel('relative angle difference [°]')
    # plt.ylabel('serial bias [°]')
    # plt.legend()
    # #plt.savefig('../Figures/Neural/SerialBias/HighLowDecoding/SerialBias_highVSlowDecoding_Sa2.png', dpi=100)
    # plt.show()
    
    # plt.figure()
    # plt.plot(df_sb_onesided_high_all.groupby('rel_loc').median()['err'], color='lightcoral', label='median,high')
    # plt.plot(df_sb_onesided_low_all.groupby('rel_loc').median()['err'], color='lightblue', label='median,low')
    # plt.show()
    
    ttest.append(ttest_ind(df_sb_onesided_high_all['err'].values, df_sb_onesided_low_all['err'].values)[0])#0.3
    pval.append(ttest_ind(df_sb_onesided_high_all['err'].values, df_sb_onesided_low_all['err'].values)[1])#0.3

#x=np.linspace(-((steps-(borders_all['Sa'][0][3]-borders_all['Sa'][0][1])-1))*50, (borders_all['Sa'][0][3]-borders_all['Sa'][0][1])*bins, steps-1)
x= np.linspace(-((borders_all['Wa'][0][8]-borders_all['Wa'][0][7])+borders_all['Wa'][0][1]-1)*50, (borders_all['Wa'][0][2]-borders_all['Wa'][0][1])*bins, (borders_all['Wa'][0][8]-borders_all['Wa'][0][7])+borders_all['Wa'][0][1]+(borders_all['Wa'][0][2]-borders_all['Wa'][0][1]))
plt.figure()
plt.subplots_adjust(wspace=0.3)
plt.subplot(121)
plt.axvline(0, color='grey', dashes=[5,5])
#plt.axvline(8*bins, color='grey', dashes=[5,5])
plt.plot(x,ttest[::-1], 'o-',color='cornflowerblue')
plt.xlabel('time to stim$_n$ start [ms]')
plt.ylabel('t-statistic')
plt.xticks([-200,0,200,400], ['-200','start','200','end'])
plt.subplot(122)
plt.axvline(0, color='grey', dashes=[5,5])
#plt.axvline(8*bins, color='grey', dashes=[5,5])
plt.plot(x,pval[::-1], 'o-', color='darkred')
plt.axhline(0.05, color='grey', dashes=[2,2])
plt.xlabel('time to stim$_n$ start [ms]')
plt.ylabel('p-value')
plt.xticks([-200,0,200,400], ['-200','start','200','end'])
#plt.savefig('../Figures/Neural/SerialBias/HighLowDecoding/pValueSerialBias_HighLowDecoding_numBinsIncluded_allMonkeys_stim.png', dpi=100)
plt.show()

print('Minimum possible p-value : '+str(np.array(pval)[np.where(np.array(pval)<0.05)[0]])+' at '+str(np.where(np.array(pval)<0.05)))


plt.figure()
plt.subplot(211)
plt.axvline(0, color='grey', dashes=[5,5])
plt.plot(x,num_high0[::-1], color='lightblue',label='0,h')
plt.plot(x,num_high45[::-1],color='deepskyblue',label='45,h')
plt.plot(x,num_high90[::-1],color='dodgerblue',label='90,h')
plt.plot(x,num_high135[::-1],color='mediumblue',label='135,h')
plt.plot(x,num_high180[::-1],color='midnightblue',label='180,h')
plt.xticks([-300,-200,-100,0,100,200,300,400], ['-300','-200','-100','start','100','200','300','end'])
plt.ylabel('# high decoding trials')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplot(212)
plt.axvline(0, color='grey', dashes=[5,5])
plt.plot(x,num_low0[::-1], color='salmon', label='0,l')
plt.plot(x,num_low45[::-1],color='lightcoral',label='45,l')
plt.plot(x,num_low90[::-1],color='indianred',label='90,l')
plt.plot(x,num_low135[::-1],color='firebrick',label='135,l')
plt.plot(x,num_low180[::-1],color='darkred',label='180,l')
plt.xlabel('time to stim$_n$ start [ms]')
plt.ylabel('# low decoding trials')
plt.xticks([-300,-200,-100,0,100,200,300,400], ['-300','-200','-100','start','100','200','300','end'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.savefig('../Figures/Neural/SerialBias/HighLowDecoding/TrialSplit_HighLowDecoding_numBinsIncluded_Sa.png', dpi=100)
plt.show()


# plt.figure()
# plt.hist(df_sb_onesided_high_all['rel_loc'], color='blue',alpha=0.2, label='high')
# plt.hist(df_sb_onesided_low_all['rel_loc'], color='red', alpha=0.2, label='low')
# plt.xlabel('relative angle difference [°]')
# plt.ylabel('# trials')
# plt.legend()
# plt.tight_layout()
# #plt.savefig('../Figures/Neural/SerialBias/HighLowDecoding/HistEndDelay_HighLowDecoding_numBinsIncluded_Sa.png', dpi=100)
# plt.show()