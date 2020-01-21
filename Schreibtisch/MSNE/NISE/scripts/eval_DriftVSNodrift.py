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
            results.append([float(s1.split(" ")[0]),float(s1.split(" ")[1]),float(s1.split(" ")[2])])
        except:
            continue
        
    results=np.array(results)
    return results

filename_reg = '/home/melanie/Schreibtisch/MSNE/NISE/results/UndisturbedBumpAveragePosition.txt'
filename_d =  '/home/melanie/Schreibtisch/MSNE/NISE/results/DisturbedBumpAveragePosition.txt'

regular = get_results(filename_reg)
disturbed = get_results(filename_d)

mean_reg = np.mean(regular[:,2])
std_reg = np.std(regular[:,2])

mean_d = np.mean(disturbed[:,2])
std_d = np.std(disturbed[:,2])


plt.rcParams.update({'font.size': 14})
plt.errorbar(['regular', 'perturbed'],[mean_reg, mean_d], [std_reg, std_d], linestyle='none', color='black', linewidth=4, Marker='o', MarkerSize='10')
plt.ylabel('Endposition in °', FontSize= 16)
plt.plot([-0.1,1.1],[regular[0,0],regular[0,0]], color='r', linestyle='-.')
plt.plot([-0.1, 1.1], [disturbed[0,1], disturbed[0,1]], color='b', linestyle = '-.')
#plt.savefig('/home/melanie/Schreibtisch/MSNE/NISE/results/Std_regularVSperturbed.png', dpi=300)