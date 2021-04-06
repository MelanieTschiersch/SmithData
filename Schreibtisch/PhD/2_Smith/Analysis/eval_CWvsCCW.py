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


file = io.open("../Results/CWvsCCW.txt", "r")#_AllMonkeys
contents = file.readlines()

values = [eval(contents[i]) for i in range(len(contents))]

sess_number = [3,1,1]
bins=50

acc_prev_cw = []
acc_curr_cw = []
acc_prev_ccw = []
acc_curr_ccw= []
helper=0
for idx, mono in enumerate(['Sa', 'Pe', 'Wa']):
    for sess in range(sess_number[idx]):
        acc_prev_cw.append(np.array(values[helper]))
        helper+=1
        acc_curr_cw.append(values[helper])
        helper+=1
        acc_prev_ccw.append(values[helper])
        helper+=1
        acc_curr_ccw.append(values[helper])
        helper+=1

trial=6
plt.figure()
plt.subplot(121)
plt.plot(acc_prev_cw[:,trial]*-1)
plt.plot(acc_prev_ccw[:,trial]*-1)
plt.subplot(122)
plt.plot(acc_curr_cw[:,trial]*-1)
plt.plot(acc_curr_ccw[:,trial]*-1)
plt.show()

########################### MEAN CW VS CCW ACTIVITY ######################################################

f, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
ax1.plot(np.mean(acc_prev_cw, axis=1)*-1)
ax1.plot(np.mean(acc_prev_ccw, axis=1)*-1)
ax2.plot(np.mean(acc_curr_cw, axis=1)*-1)
ax2.plot(np.mean(acc_curr_ccw, axis=1)*-1)
plt.show()

########################### HIGH VS LOW DECODING DELAYS ######################################################
