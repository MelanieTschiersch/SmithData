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

filename_reg = 'C:\\Users\\Maria\\Desktop\\Norepinephrin3101_Daichi_deg45_slowerwashout2.txt'

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

daichi_res_learn = np.arctan(np.array([0.0263,0.0054, 0.0076, 0.0039, 0.0011, 0.0008, 0.0031, 0.0015, 0.0019, 0.0006, 0.0047, -0.001, 0.0038, 0.0029, 0.0022, -0.0008, 0.0009, 0.0004, -0.0003, 0.001, 0.001, -0.0006, -0.0005, -0.0003,-0.0024, -0.0028, -0.0024, 0.0023, -0.0027, 0.0003, -0.0021, -0.0013, -0.0025, -0.0006, -0.0003, 0.0011, -0.0036, -0.0005, -0.0014, -0.0022])/0.1)*180/np.pi
daichi_res_wo = np.arctan(np.array([-0.0042,-0.0293, -0.0213, -0.0247, -0.0229, -0.0201, -0.0237, -0.0234, -0.0238, -0.0262, -0.0237, -0.0229, -0.017, -0.0133, -0.0112, -0.0108, -0.0083, -0.003, -0.0074, -0.0018, -0.0048, -0.0055, -0.0027, -0.004, -0.0057, -0.0046, -0.002, -0.0047, -0.0033, -0.0043, -0.0064])/0.1)*180/np.pi

daichi_res_std_learn = np.arctan(np.array([0.0077, 0.0134, 0.0111, 0.0082, 0.01, 0.0075, 0.0058, 0.0065, 0.005, 0.0081, 0.0071, 0.0102, 0.0085, 0.0058, 0.0061, 0.0019, 0.0045, 0.0036, 0.005, 0.0068, 0.0093, 0.0049, 0.008, 0.005, 0.004, 0.009, 0.0041, 0.0057, 0.0042, 0.0079, 0.0065, 0.0047, 0.0044, 0.0069, 0.0037, 0.006, 0.0043, 0.0077, 0.0036, 0.0035])/0.1)*180/np.pi
daichi_res_std_wo = np.arctan(np.array([0.0033, 0.0055, 0.0091, 0.007, 0.0091, 0.0071, 0.0077, 0.0087, 0.0071, 0.0111, 0.0058, 0.0056, 0.0086, 0.0082, 0.0078, 0.0052, 0.0057, 0.0051, 0.006, 0.0043, 0.0058, 0.0054, 0.0052, 0.0041, 0.0031, 0.0036, 0.0044, 0.0036, 0.0049, 0.0061, 0.0056])/0.1)*180/np.pi

daichi_res = np.concatenate((daichi_res_learn, daichi_res_wo))
daichi_std = np.concatenate((daichi_res_std_learn, daichi_res_std_wo))

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.errorbar(np.linspace(1, int(trialmax), int(trialmax)),mean_trial[:,2], std_trial[:,2], linestyle='none', color='darkblue', linewidth=4, Marker='o', MarkerSize='10', label = 'our data')
plt.errorbar(np.linspace(1, int(trialmax), int(trialmax)),daichi_res, daichi_std, linestyle='none', color='g', linewidth=4, Marker='o', MarkerSize='10', label = 'daichi')
plt.axhline(y=0, linestyle='--', color='k')
plt.ylabel('Error in endposition [°]', FontSize= 16)
plt.xlabel('# of trials', FontSize= 16)
plt.legend()
#plt.savefig('Error_endpos_Daichi.png', bbox_inches='tight', dpi=500)
#plt.savefig('/home/melanie/Schreibtisch/MSNE/NISE/results/Adaptation_Average1SetLearning2.png', dpi=300)

plt.figure()
plt.plot(np.linspace(1, int(trialmax), int(trialmax)), mean_trial[:,3], '.-')
plt.plot(np.linspace(1, int(trialmax), int(trialmax)), mean_trial[:,4], '.-')
plt.plot(np.linspace(1, int(trialmax), int(trialmax)), mean_trial[:,5], '.-')
plt.ylabel('NMDA conductance [nS]', FontSize= 16)
plt.xlabel('# of trials', FontSize= 16)
#plt.savefig('NMDAconductance_endpos_Daichi.png', bbox_inches='tight', dpi=500)
