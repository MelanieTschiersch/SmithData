#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:12:12 2020

@author: melanie
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import ast
import numpy as np
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

def read_baseline(filename, mode):
    ''' 
    reads in accuracies/std for 1000x mixed labels per timepoint for r² and MSE evaluation
    INPUT:
        filename:       name of file containing 1000x croxxvalidation on mixed labels
    OUTPUT:
        r2_baseline:    previous accuracy, std and current accuracy, std for the mixed labels R² evaluation
        mse_baseline:   previous accuracy, std and current accuracy, std for the mse evaluation turned into MSE accuracy[%]
    
    '''
    file = io.open(filename, "r")
    contents = file.readlines()

    values = [eval(contents[i]) for i in range(len(contents))]

    if mode == 'r2':
        r2_baseline = {'acc_prev':values[0], 'std_prev':values[1], 'acc_curr':values[2], 'std_curr':values[3]}
        return r2_baseline
        
    elif mode =='MSE':
        #values[4] = 100-np.array(values[4])*100/circdist(0, np.pi)[0]**2#np.pi
        #values[5] = np.array(values[5])*100
        #values[6] = 100-np.array(values[6])*100/circdist(0, np.pi)[0]**2#np.pi
        #values[7] = np.array(values[7])*100
        mse_baseline = {'acc_prev':values[4], 'std_prev':values[5], 'acc_curr':values[6], 'std_curr':values[7]}
        return mse_baseline

    return print('ERROR: mode must either be r2 or MSE.')


def plot_hemispheres_delayVSresponse(acc_bias_prev_delay_ipsi, std_bias_prev_delay_ipsi,\
                               acc_bias_curr_delay_ipsi, std_bias_curr_delay_ipsi, label_delay_ipsi,\
                                 acc_bias_prev_response_ipsi, std_bias_prev_response_ipsi,\
                               acc_bias_curr_response_ipsi, std_bias_curr_response_ipsi,label_response_ipsi,\
                                 acc_bias_prev_delay_contra, std_bias_prev_delay_contra,\
                               acc_bias_curr_delay_contra, std_bias_curr_delay_contra, label_delay_contra,\
                                 acc_bias_prev_response_contra, std_bias_prev_response_contra,\
                               acc_bias_curr_response_contra, std_bias_curr_response_contra,label_response_contra,\
                                 borders, borders_pastdelay, borders_full, mode, baseline, title):
    
    if mode =='MSE':
        acc_bias_prev_delay_ipsi = np.array(acc_bias_prev_delay_ipsi)*-1#100-100/(0.5*np.pi)*np.array(acc_bias_prev_ipsi)
        acc_bias_curr_delay_ipsi = np.array(acc_bias_curr_delay_ipsi)*-1#100-100/(0.5*np.pi)*np.array(acc_bias_curr_ipsi)
        acc_bias_prev_response_ipsi = np.array(acc_bias_prev_response_ipsi)*-1#100-100/(0.5*np.pi)*np.array(acc_bias_prev_ipsi)
        acc_bias_curr_response_ipsi = np.array(acc_bias_curr_response_ipsi)*-1#100-100/(0.5*np.pi)*np.array(acc_bias_curr_ipsi)
        acc_bias_prev_delay_contra = np.array(acc_bias_prev_delay_contra)*-1#100-100/(0.5*np.pi)*np.array(acc_bias_prev_contra)
        acc_bias_curr_delay_contra = np.array(acc_bias_curr_delay_contra)*-1#100-100/(0.5*np.pi)*np.array(acc_bias_curr_contra)
        acc_bias_prev_response_contra = np.array(acc_bias_prev_response_contra)*-1#100-100/(0.5*np.pi)*np.array(acc_bias_prev_contra)
        acc_bias_curr_response_contra = np.array(acc_bias_curr_response_contra)*-1
        
        baseline['acc_prev'] = np.array(baseline['acc_prev'])*-1
        baseline['acc_curr'] = np.array(baseline['acc_curr'])*-1
    elif mode=='r2':
        acc_bias_prev_delay_ipsi = acc_bias_prev_delay_ipsi
    else:
        print('Mode needs to be either:\n'+'r2 for evaluation with r² metric\n'+\
                      'or\nMSE for evaluation with mean-squared error.')
    
    c_base = 'grey'
    
    #x = np.linspace(0,len(acc_bias_prev_ipsi[:borders_full[-2]])*bins, len(acc_bias_prev_ipsi[:borders[-2]]))
    x = np.linspace(0,borders_full[3]*bins, borders_full[3])
    #x2 = np.linspace(-(borders_pastdelay[2]-borders[-2])*bins,(borders_pastdelay[3]-borders_pastdelay[2])*bins, len(acc_bias_prev_ipsi[borders[-2]:borders[-1]]))
    x2 = np.linspace((borders_full[3]-borders_full[4])*bins,(borders_full[7]-borders_full[4])*bins, borders_full[7]-borders_full[3])
    #x3 = np.linspace(-(len(acc_bias_prev_ipsi[borders[-1]:]))*bins,0, len(acc_bias_prev_ipsi[borders[-1]:]))
    x3 = np.linspace((borders_full[7]-borders_full[8])*bins,0, borders_full[8]-borders_full[7])
    x4 = np.linspace(0,borders_full[3]*bins, borders_full[3])
    labels = np.array(['0', '$S_{n-1}$', '$SE_{n-1}$', '$D_{n-1}$', '$D_{n-1}$','$S_{n-1}$','$R_{n-1}$', '$E_{n-1}$'])
    labels_curr = np.array(['$0_n$', '$S_n$', '$SE_n$'])

    y_low=-1.1
    y_high = 0.8
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True, figsize=(18,8))
    plt.subplots_adjust(wspace=0.05)
    ax1.plot(x,acc_bias_prev_delay_contra[:borders_full[3]], color='darkred')
    ax1.plot(x,acc_bias_prev_delay_ipsi[:borders_full[3]], color='lightcoral')
    ax1.plot(x,acc_bias_prev_response_contra[:borders_full[3]], color='darkgreen')
    ax1.plot(x,acc_bias_prev_response_ipsi[:borders_full[3]], color='limegreen')
    # errorbar
    ax1.fill_between(x, acc_bias_prev_delay_contra[:borders[-2]]-0.5*np.array(std_bias_prev_delay_contra[:borders[-2]]), acc_bias_prev_delay_contra[:borders[-2]]+0.5*np.array(std_bias_prev_delay_contra[:borders[-2]]), color='darkred', alpha=0.2)
    ax1.fill_between(x, acc_bias_prev_delay_ipsi[:borders[-2]]-0.5*np.array(std_bias_prev_delay_ipsi[:borders[-2]]), acc_bias_prev_delay_ipsi[:borders[-2]]+0.5*np.array(std_bias_prev_delay_ipsi[:borders[-2]]), color='lightcoral', alpha=0.2)
    ax1.fill_between(x, acc_bias_prev_response_contra[:borders[-2]]-0.5*np.array(std_bias_prev_response_contra[:borders[-2]]), acc_bias_prev_response_contra[:borders[-2]]+0.5*np.array(std_bias_prev_response_contra[:borders[-2]]), color='darkgreen', alpha=0.2)
    ax1.fill_between(x, acc_bias_prev_response_ipsi[:borders[-2]]-0.5*np.array(std_bias_prev_response_ipsi[:borders[-2]]), acc_bias_prev_response_ipsi[:borders[-2]]+0.5*np.array(std_bias_prev_response_ipsi[:borders[-2]]), color='limegreen', alpha=0.2)
    # plot baseline
    #ax1.axhline(np.mean(baseline['acc_prev']), *ax1.get_xlim(), color='k')
    #ax1.plot(x,baseline['acc_prev'][:borders[-2]], color=c_base)
    #ax1.fill_between(x, baseline['acc_prev'][:borders[-2]]-0.5*np.array(baseline['std_prev'][:borders[-2]]), baseline['acc_prev'][:borders[-2]]+0.5*np.array(baseline['std_prev'][:borders[-2]]), color=c_base, alpha=0.2)

    # mark training time of delay decoder
    #ax1.plot([borders_full[2]*bins,borders_full[3]*bins], [-0.1, -0.1], color='darkgreen', linewidth=6)
    ax1.set_xlabel('$start_{n-1}$ aligned [ms]', fontsize=14)
    if mode =='MSE':
        ax1.set_ylabel('inverted '+str(mode)+' (k=10)', fontsize=14)#avg $r^2$ score
    else:
        ax1.set_ylabel(str(mode)+' (k=10)', fontsize=14)#avg $r^2$ score
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')


    ax2.plot(x2,acc_bias_prev_delay_contra[borders_full[3]:borders_full[7]], color='darkred')
    ax2.plot(x2,acc_bias_prev_delay_ipsi[borders_full[3]:borders_full[7]], color='lightcoral')
    ax2.plot(x2,acc_bias_prev_response_contra[borders_full[3]:borders_full[7]], color='darkgreen')
    ax2.plot(x2,acc_bias_prev_response_ipsi[borders_full[3]:borders_full[7]], color='limegreen')
    # errorbar
    ax2.fill_between(x2, acc_bias_prev_delay_contra[borders_full[3]:borders_full[7]]-0.5*np.array(std_bias_prev_delay_contra[borders_full[3]:borders_full[7]]), acc_bias_prev_delay_contra[borders_full[3]:borders_full[7]]+0.5*np.array(std_bias_prev_delay_contra[borders_full[3]:borders_full[7]]), color='darkred', alpha=0.2)
    ax2.fill_between(x2, acc_bias_prev_delay_ipsi[borders_full[3]:borders_full[7]]-0.5*np.array(std_bias_prev_delay_ipsi[borders_full[3]:borders_full[7]]), acc_bias_prev_delay_ipsi[borders_full[3]:borders_full[7]]+0.5*np.array(std_bias_prev_delay_ipsi[borders_full[3]:borders_full[7]]), color='lightcoral', alpha=0.2)
    ax2.fill_between(x2, acc_bias_prev_response_contra[borders_full[3]:borders_full[7]]-0.5*np.array(std_bias_prev_response_contra[borders_full[3]:borders_full[7]]), acc_bias_prev_response_contra[borders_full[3]:borders_full[7]]+0.5*np.array(std_bias_prev_response_contra[borders_full[3]:borders_full[7]]), color='darkgreen', alpha=0.2)
    ax2.fill_between(x2, acc_bias_prev_response_ipsi[borders_full[3]:borders_full[7]]-0.5*np.array(std_bias_prev_response_ipsi[borders_full[3]:borders_full[7]]), acc_bias_prev_response_ipsi[borders_full[3]:borders_full[7]]+0.5*np.array(std_bias_prev_response_ipsi[borders_full[3]:borders_full[7]]), color='limegreen', alpha=0.2)
    # plot baseline
    #ax2.axhline(np.mean(baseline['acc_prev']), *ax2.get_xlim(), color='k')
    #ax2.plot(x2,baseline['acc_prev'][borders[-2]:borders[-1]], color=c_base)
    #ax2.fill_between(x2, baseline['acc_prev'][borders[-2]:borders[-1]]-0.5*np.array(baseline['std_prev'][borders[-2]:borders[-1]]), baseline['acc_prev'][borders[-2]:borders[-1]]+0.5*np.array(baseline['std_prev'][borders[-2]:borders[-1]]), color=c_base, alpha=0.2)

    # mark training period of response decoder
    #ax2.plot([(borders_full[6]-borders_full[5])*bins,(borders_full[7]-borders_full[5])*bins], [-0.1, -0.1], color='darkorange', linewidth=4)
    ax2.set_xlabel('$report_{n-1}$ aligned [ms]', fontsize=14)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    
    ax3.plot(x3,acc_bias_prev_delay_contra[borders_full[7]:], color='darkred')
    ax3.plot(x3,acc_bias_prev_delay_ipsi[borders_full[7]:], color='lightcoral')
    ax3.plot(x3,acc_bias_prev_response_contra[borders_full[7]:], color='darkgreen')
    ax3.plot(x3,acc_bias_prev_response_ipsi[borders_full[7]:], color='limegreen')
    ax3.fill_between(x3, acc_bias_prev_delay_contra[borders_full[7]:]-0.5*np.array(std_bias_prev_delay_contra[borders_full[7]:]), acc_bias_prev_delay_contra[borders_full[7]:]+0.5*np.array(std_bias_prev_delay_contra[borders_full[7]:]), color='darkred', alpha=0.2)
    ax3.fill_between(x3, acc_bias_prev_delay_ipsi[borders_full[7]:]-0.5*np.array(std_bias_prev_delay_ipsi[borders_full[7]:]), acc_bias_prev_delay_ipsi[borders_full[7]:]+0.5*np.array(std_bias_prev_delay_ipsi[borders_full[7]:]), color='lightcoral', alpha=0.2)
    ax3.fill_between(x3, acc_bias_prev_response_contra[borders_full[7]:]-0.5*np.array(std_bias_prev_response_contra[borders_full[7]:]), acc_bias_prev_response_contra[borders_full[7]:]+0.5*np.array(std_bias_prev_response_contra[borders_full[7]:]), color='darkgreen', alpha=0.2)
    ax3.fill_between(x3, acc_bias_prev_response_ipsi[borders_full[7]:]-0.5*np.array(std_bias_prev_response_ipsi[borders_full[7]:]), acc_bias_prev_response_ipsi[borders_full[7]:]+0.5*np.array(std_bias_prev_response_ipsi[borders_full[7]:]), color='limegreen', alpha=0.2)
    # plot baseline
    #ax3.axhline(np.mean(baseline['acc_prev']), *ax3.get_xlim(), color='k')
    #ax3.plot(x3,baseline['acc_prev'][borders[-1]:], color=c_base)
    #ax3.fill_between(x3, baseline['acc_prev'][borders[-1]:]-0.5*np.array(baseline['std_prev'][borders[-1]:]), baseline['acc_prev'][borders[-1]:]+0.5*np.array(baseline['std_prev'][borders[-1]:]), color=c_base, alpha=0.2)
    #for period in range(len(borders_pastdelay)):
    #    ax2.axvline((borders_pastdelay[period]-borders[-1])*bins, *ax.get_xlim(), color='grey',dashes=[4,2])
    #ax2.set_xticks(bins*np.array(borders_pastdelay-borders[-1]))
    #ax2.set_xticklabels(labels[len(borders):], fontsize=12)
    #ax2.set_yticklabels(fontsize=12)
    ax3.set_xlabel('$end_{n-1}$ aligned [ms]', fontsize=14)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.xaxis.set_ticks_position('bottom')

    ax4.plot(x4,acc_bias_curr_delay_contra[:borders_full[3]], color='darkred', label = label_delay_contra)
    ax4.plot(x4,acc_bias_curr_delay_ipsi[:borders_full[3]], color='lightcoral', label = label_delay_ipsi)
    ax4.plot(x4,acc_bias_curr_response_contra[:borders_full[3]], color='darkgreen', label = label_response_contra)
    ax4.plot(x4,acc_bias_curr_response_ipsi[:borders_full[3]], color='limegreen', label = label_response_ipsi)
    # errorbar
    ax4.fill_between(x4, acc_bias_curr_delay_contra[:borders_full[3]]-0.5*np.array(std_bias_curr_delay_contra[:borders_full[3]]), acc_bias_curr_delay_contra[:borders_full[3]]+0.5*np.array(std_bias_curr_delay_contra[:borders_full[3]]), color='darkred', alpha=0.2)
    ax4.fill_between(x4, acc_bias_curr_delay_ipsi[:borders_full[3]]-0.5*np.array(std_bias_curr_delay_ipsi[:borders_full[3]]), acc_bias_curr_delay_ipsi[:borders_full[3]]+0.5*np.array(std_bias_curr_delay_ipsi[:borders_full[3]]), color='lightcoral', alpha=0.2)
    ax4.fill_between(x4, acc_bias_curr_response_contra[:borders_full[3]]-0.5*np.array(std_bias_curr_response_contra[:borders_full[3]]), acc_bias_curr_response_contra[:borders_full[3]]+0.5*np.array(std_bias_curr_response_contra[:borders_full[3]]), color='darkgreen', alpha=0.2)
    ax4.fill_between(x4, acc_bias_curr_response_ipsi[:borders_full[3]]-0.5*np.array(std_bias_curr_response_ipsi[:borders_full[3]]), acc_bias_curr_response_ipsi[:borders_full[3]]+0.5*np.array(std_bias_curr_response_ipsi[:borders_full[3]]), color='limegreen', alpha=0.2)
    # plot baseline
    #ax4.axhline(np.mean(baseline['acc_prev']), *ax4.get_xlim(), color='k')
    #ax4.plot(x4,baseline['acc_curr'][:borders[-3]], color=c_base, label='baseline')
    #ax4.fill_between(x4, baseline['acc_curr'][:borders[-3]]-0.5*np.array(baseline['std_curr'][:borders[-3]]), baseline['acc_curr'][:borders[-3]]+0.5*np.array(baseline['std_curr'][:borders[-3]]), color=c_base, alpha=0.2)
    #for period in range(len(borders[:3])):
    #    ax3.axvline(borders[period]*bins, *ax.get_ylim(), color='grey',dashes=[4,2])
    #ax3.set_xticks(bins*np.array(borders[:3]))
    #ax3.set_xticklabels(labels_curr, fontsize=12)
    ax4.set_xlabel('$start_n$ aligned [ms]', fontsize=14)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.xaxis.set_ticks_position('bottom')

    y0=ax2.get_ylim()[0]
    y1=ax2.get_ylim()[1]
    ax1.fill_between([borders_full[1]*bins, borders_full[2]*bins], y0, y1, color='red', alpha=0.2)
    ax2.fill_between([0, (borders_full[5]-borders_full[6])*bins], y0,y1, color='grey', alpha=0.2)
    ax4.fill_between([borders_full[1]*bins, borders_full[2]*bins], y0,y1, color='red', alpha=0.2)

    plt.legend(fontsize=16)

    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.title(title, fontsize=15) 

    plt.savefig('../Figures/Neural/SerialBias/hemispheres/AllMonkeys_leftright/IpsiContraContinuous_AllMonkeys_'+title+'.png', dpi=100)
    plt.show()
    
    if mode =='MSE':
        baseline['acc_prev'] = np.array(baseline['acc_prev'])*-1
        baseline['acc_curr'] = np.array(baseline['acc_curr'])*-1
    return

##########################################################################
#                              LOAD DATA                                 #
##########################################################################
    
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


sess_number = [3,1,1]
helper=0
for idx, mono in enumerate(['Sa', 'Pe', 'Wa']):
    for sess in range(sess_number[idx]):
        #only use Sa, sess0
        df_Sa0_monkey = mono
        df_Sa0_sess = sess
        df_Sa0 = df_dat.loc[(df_dat['monkey']==df_Sa0_monkey) & (df_dat['session']==df_Sa0_sess)]
        
        # make spike trains into csr matrix for each trial
        mat = [csr_matrix(df_Sa0.loc[n,'sp_train']) for n in df_Sa0['sp_train'].index]
        df_Sa0.loc[:,'n_mat'] = mat
        
        ##########################################################################
        #                               SORT DATA                                #
        ##########################################################################
        
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
        
        ##########################################################################
        #                         LOAD IPSI/CONTRA                               #
        ##########################################################################
            
        file = io.open("../Results/IpsiContraContinuous_AllMonkeys.txt", "r")
        contents = file.readlines()
        
        values = [eval(contents[i]) for i in range(len(contents))]
        
        acc_prev_delay_ipsi = []
        std_prev_delay_ipsi = []
        acc_curr_delay_ipsi = []
        std_curr_delay_ipsi = []
        acc_prev_response_ipsi = []
        std_prev_response_ipsi = []
        acc_curr_response_ipsi = []
        std_curr_response_ipsi = []
        acc_prev_delay_contra = []
        std_prev_delay_contra = []
        acc_curr_delay_contra = []
        std_curr_delay_contra = []
        acc_prev_response_contra = []
        std_prev_response_contra = []
        acc_curr_response_contra = []
        std_curr_response_contra = []
        

        for leftright in range(2): # mix left VS right neurons
            acc_prev_delay_ipsi.append(np.array(values[helper]))
            helper+=1
            std_prev_delay_ipsi.append(values[helper])
            helper+=1
            acc_curr_delay_ipsi.append(values[helper])
            helper+=1
            std_curr_delay_ipsi.append(values[helper])
            helper+=1
            acc_prev_response_ipsi.append(np.array(values[helper]))
            helper+=1
            std_prev_response_ipsi.append(values[helper])
            helper+=1
            acc_curr_response_ipsi.append(values[helper])
            helper+=1
            std_curr_response_ipsi.append(values[helper])
            helper+=1
            acc_prev_delay_contra.append(np.array(values[helper]))
            helper+=1
            std_prev_delay_contra.append(values[helper])
            helper+=1
            acc_curr_delay_contra.append(values[helper])
            helper+=1
            std_curr_delay_contra.append(values[helper])
            helper+=1
            acc_prev_response_contra.append(np.array(values[helper]))
            helper+=1
            std_prev_response_contra.append(values[helper])
            helper+=1
            acc_curr_response_contra.append(values[helper])
            helper+=1
            std_curr_response_contra.append(values[helper])
            helper+=1
        
############# PLOT IPSI/CONTRA DELAY/ITI decoder across all monkeys, all sessions######################     


            label_delay_ipsi='delay,ipsi'
            label_response_ipsi='ITI,ipsi'
            label_delay_contra='delay,contra'
            label_response_contra='ITI,contra'
            mode='MSE'
            baseline_file = "../Results/baselineAcc.txt"
            baseline = read_baseline(baseline_file, mode)
            
            title = str(mono)+str(sess)
            plot_hemispheres_delayVSresponse(acc_prev_delay_ipsi[leftright], std_prev_delay_ipsi[leftright],\
                                         acc_curr_delay_ipsi[leftright], std_curr_delay_ipsi[leftright], label_delay_ipsi,\
                                             acc_prev_response_ipsi[leftright], std_prev_response_ipsi[leftright],\
                                                 acc_curr_response_ipsi[leftright], std_curr_response_ipsi[leftright],label_response_ipsi,\
                                             acc_prev_delay_contra[leftright], std_prev_delay_contra[leftright],\
                                           acc_curr_delay_contra[leftright], std_curr_delay_contra[leftright], label_delay_contra,\
                                             acc_prev_response_contra[leftright], std_prev_response_contra[leftright],\
                                           acc_curr_response_contra[leftright], std_curr_response_contra[leftright],label_response_contra,\
                                             borders, borders_pastdelay, borders_full, mode, baseline, title)
            
            
             # plot ipsi, contra delay response decoder for left and right neurons combined
        #title = str(mono)+str(sess)
        #plot_hemispheres_delayVSresponse(np.mean(acc_prev_delay_ipsi, axis=0), np.mean(std_prev_delay_ipsi, axis=0),\
        #                                 np.mean(acc_curr_delay_ipsi, axis=0), np.mean(std_curr_delay_ipsi, axis=0), label_delay_ipsi,\
        #                                     np.mean(acc_prev_response_ipsi, axis=0), np.mean(std_prev_response_ipsi, axis=0),\
        #                                         np.mean(acc_curr_response_ipsi, axis=0), np.mean(std_curr_response_ipsi, axis=0),label_response_ipsi,\
        #                                     np.mean(acc_prev_delay_contra, axis=0), np.mean(std_prev_delay_contra, axis=0),\
        #                                   np.mean(acc_curr_delay_contra, axis=0), np.mean(std_curr_delay_contra, axis=0), label_delay_contra,\
        #                                     np.mean(acc_prev_response_contra, axis=0), np.mean(std_prev_response_contra, axis=0),\
        #                                   np.mean(acc_curr_response_contra, axis=0), np.mean(std_curr_response_contra, axis=0),label_response_contra,\
        #                                     borders, borders_pastdelay, borders_full, mode, baseline, title)
            
            
           

