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

filename_reg = '/home/melanie/Schreibtisch/MSNE/NISE/results/SeveralBumps.txt'

results = get_results(filename_reg)

plt.rcParams.update({'font.size': 14})
plt.plot(np.linspace(1, len(results), len(results)), results[:,2], linestyle='none', color='darkblue', linewidth=4, Marker='o', MarkerSize='10')
plt.axhline(y=0, linestyle='--', color='k')
plt.ylabel('Error in endposition °', FontSize= 16)
plt.xlabel('# of trial', FontSize= 16)
#plt.savefig('/home/melanie/Schreibtisch/MSNE/NISE/results/Adaptation_Force2Learning1.png', dpi=300)