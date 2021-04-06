#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:48:12 2019

@author: melanie

Evaluates percentage of unconcscious trials, for 1 Stimulus (no ITI) and varying delay 2.5s,4s
"""

import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#from circ_stats import *
from mpl_toolkits.mplot3d import Axes3D

#plt.style.use('/home/melanie/internship/PaperDoubleFig.mplstyle')
#style.use('classic')
#sns.set_context('talk')

def get_results(files):
    """
    Gets results from a file for perturbed, non-perturbed motor trials
    Input:  filname_reg = path of unperturbed result file   n x 3   [cue angle, distractor angle, final bump position]
            filename_d  = path to perturbed results file    n x 3   [cue angle, distractor angle, final bump position]
    Output: 
            results of specified input
    """
    f=open(files)
    s=f.readlines()
    results=[]
    for i,s1 in enumerate(s):
        try:
            results.append([float(s1.split(" ")[0]),float(s1.split(" ")[1]),float(s1.split(" ")[2]),float(s1.split(" ")[3]),float(s1.split(" ")[4]),float(s1.split(" ")[5]),float(s1.split(" ")[6]),float(s1.split(" ")[7]),float(s1.split(" ")[8])])
        except:
            continue
        
    results=np.array(results)
    return results

filename_reg = '/home/melanie/Schreibtisch/MSNE/NISE/results/Norepinephrin3001_shortDaichi.txt'

results = get_results(filename_reg)
#print(results)
results[:,2] = results[:,2]-180
setmax = results[0,7]
trialmax = results[0,8]

ordered_res = np.zeros((int(setmax), int(trialmax), 9))
for sets in range(0,int(setmax)):
    ordered_res[sets, :, :] = results[int(sets)*int(trialmax):int(sets*trialmax+trialmax),:]

mean_trial = np.zeros((int(trialmax),9))
std_trial = np.zeros((int(trialmax),9))
for trial in range(0, int(trialmax)):
    mean_trial[trial,:] = np.mean(ordered_res[:,trial,:],axis=0)
    std_trial[trial,:] = np.std(ordered_res[:,trial,:],axis=0)


plt.figure()
plt.rcParams.update({'font.size': 14})
plt.errorbar(np.linspace(1, int(trialmax), int(trialmax)),mean_trial[:,2], std_trial[:,2], linestyle='none', color='darkblue', linewidth=4, Marker='o', MarkerSize='10')
plt.axhline(y=0, linestyle='--', color='k')
plt.ylabel('Error in endposition [°]', FontSize= 16)
plt.xlabel('# of trials', FontSize= 16)
plt.savefig('Error_endpos_Daichi.png', bbox_inches='tight', dpi=500)
#plt.savefig('/home/melanie/Schreibtisch/MSNE/NISE/results/Adaptation_Average1SetLearning2.png', dpi=300)

plt.figure()
plt.plot(np.linspace(1, int(trialmax), int(trialmax)), mean_trial[:,3], '.-')
#plt.plot(np.linspace(1, int(trialmax), int(trialmax)), mean_trial[:,4], '.-')
plt.plot(np.linspace(1, int(trialmax), int(trialmax)), mean_trial[:,5], '.-')
plt.ylabel('NMDA conductance [nS]', FontSize= 16)
plt.xlabel('# of trials', FontSize= 16)
plt.savefig('NMDAconductance_endpos_Daichi.png', bbox_inches='tight', dpi=500)
