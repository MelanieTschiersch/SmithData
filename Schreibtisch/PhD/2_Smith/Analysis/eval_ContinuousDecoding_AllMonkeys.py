#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:12:12 2020

@author: melanie
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import ast


file = io.open("../Results/ContinuousDecoding_AllMonkeys.txt", "r")
contents = file.readlines()

values = [eval(contents[i]) for i in range(len(contents))]


sess_number = [3,1,1]
bins=50

acc_prev = []
std_prev = []
acc_curr = []
std_curr= []
helper=0
for idx, mono in enumerate(['Sa', 'Pe', 'Wa']):
    for sess in range(sess_number[idx]):
        acc_prev.append(np.array(values[helper]))
        helper+=1
        std_prev.append(values[helper])
        helper+=1
        acc_curr.append(values[helper])
        helper+=1
        std_curr.append(values[helper])
        helper+=1
        
############# PLOT ALL MONKEYS ALL SESSIONS ##########################################################     
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 2,figsize=(15,15),sharey=True)
plt.subplots_adjust(wspace=0.05, hspace=0.55)
axes=[ax1, ax2, ax3, ax4, ax5]
titles=['Sa','Sa','Sa','Pe','Wa']
for i in range(5):
    x = np.linspace(0,len(acc_prev[i])*bins, len(acc_prev[i]))
    axes[i][0].plot(x,np.array(acc_prev[i])*-1, color='k')
    axes[i][0].fill_between(x,np.array(acc_prev[i])*-1-0.5*np.array(std_prev[i]), np.array(acc_prev[i])*-1+0.5*np.array(std_prev[i]),color='k', alpha=0.3)
    axes[i][0].set_title(titles[i], fontsize=15)
    axes[i][1].plot(x,np.array(acc_curr[i])*-1, color='k')
    axes[i][1].fill_between(x,np.array(acc_curr[i])*-1-0.5*np.array(std_curr[i]), np.array(acc_curr[i])*-1+0.5*np.array(std_curr[i]), color='k', alpha=0.3)

axes[i][0].set_xlabel('$start_{n-1}$ aligned [ms]', fontsize=18)
axes[i][1].set_xlabel('$start_n$ aligned [ms]', fontsize=18)
f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel('inverted MSE (k=10)', fontsize=18)
plt.tight_layout()
#plt.savefig('/home/melanie/Schreibtisch/PhD/2_Smith/Figures/Neural/SerialBias/AllMonkeys/ContinuousDecoderAllMonkeys.png', dpi=200)
plt.show()


################ PLOT  MEAN ACROSS REACTIVATION PERIOD ###########
f, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
plt.subplots_adjust(wspace=0.1)
val=30
x0 =np.linspace(-val*bins,0, val)
x = np.linspace(0,val*bins, val)
ax1.plot(x0, np.mean([acc_prev[n][-val:] for n in range(len(acc_prev))], axis=0)*-1, color='k')
ax1.fill_between(x0, np.mean([acc_prev[n][-val:] for n in range(len(acc_prev))], axis=0)*-1-\
                 0.5*np.std([acc_prev[n][-val:] for n in range(len(acc_prev))], axis=0),\
                     np.mean([acc_prev[n][-val:] for n in range(len(acc_prev))], axis=0)*-1+\
                         0.5*np.std([acc_prev[n][-val:] for n in range(len(acc_prev))], axis=0),color='k', alpha=0.3)
ax2.plot(x, np.mean([acc_curr[n][0:val] for n in range(len(acc_curr))], axis=0)*-1, color='k')
ax2.fill_between(x, np.mean([acc_curr[n][0:val] for n in range(len(acc_curr))], axis=0)*-1-\
                 0.5*np.std([acc_curr[n][0:val] for n in range(len(acc_curr))], axis=0),\
                     np.mean([acc_curr[n][0:val] for n in range(len(acc_curr))], axis=0)*-1+\
                         0.5*np.std([acc_curr[n][0:val] for n in range(len(acc_curr))], axis=0),color='k', alpha=0.3)
ax1.set_xlabel('$end_n$ aligned [ms]')
ax2.set_xlabel('$start_n$ aligned [ms]')
ax1.set_ylabel('inverted MSE (k=10)')
plt.tight_layout()
#plt.savefig('/home/melanie/Schreibtisch/PhD/2_Smith/Figures/Neural/SerialBias/AllMonkeys/Reactivations_AverageAllMonkeys.png', dpi=200)
plt.show()



