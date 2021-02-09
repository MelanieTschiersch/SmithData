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



file = io.open("../Results/baselineAcc.txt", "r")
contents = file.readlines()

values = [eval(contents[i]) for i in range(len(contents))]

r2_acc_prev = values[0]
r2_std_prev = values[1]
r2_acc_curr = values[2]
r2_std_curr = values[3]

mse_acc_prev = values[4]
mse_std_prev = values[5]
mse_acc_curr = values[6]
mse_std_curr = values[7]

x=np.linspace(0,len(r2_acc_prev), len(r2_acc_prev))
plt.figure()
plt.plot(r2_acc_prev)
plt.fill_between(x, r2_acc_prev-0.5*np.array(r2_std_prev), r2_acc_prev+0.5*np.array(r2_std_prev), alpha=0.2)
plt.show()

plt.figure()
plt.plot(mse_acc_curr)
plt.fill_between(x, mse_acc_prev-0.5*np.array(mse_std_prev), mse_acc_prev+0.5*np.array(mse_std_prev), alpha=0.2)
plt.show()