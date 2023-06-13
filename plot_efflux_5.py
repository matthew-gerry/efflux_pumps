'''
plot_efflux_5.py

Plotting functions for the five-state model of bacterial efflux pumps.

When calling functions, an object of the class Params3 should be passed to as 
argument denoted param (three-state and five-state models share a Parameters class).

Matthew Gerry, June 2023
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump


#### FUNCTIONS ####

def plot_efflux_vs_KD(param, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc, cpp_vals):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY '''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        mean_output = np.vectorize(pump.efflux_numerical_5)(param, KD_axis, Kp, KD_ratio*KD_axis, Kp_ratio*Kp, V_base, kappa, cDc, cpp)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    KD_axis_uM = 1e6*KD_axis # KD_axis in uM
    cpp_vals_uM = [1e6*x for x in cpp_vals]
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1
    for i in range(len(cpp_vals)):
        plt.semilogx(KD_axis_uM, mean_efflux[i], label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel("$J\:(s^{-1})$")
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    plt.legend()
    plt.show()


#### GLOBAL VARIABLES ####

ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting


# Parameter values
rD = 1e7 # 1/s
rp = 1e9 # 1/s
tau_C = 1e-9 # s, timescale for conformational changes
rt = 1/tau_C # 1/s, unlike in the three-state model, rt does not depend on other char. rates
vD = 1 # 1/M
vp = 0.1 # 1/M
cDo = 1e-5 # M
cpc = 1e-7 # M


# Variables
Kp = 1e-6 # M, proton binding affinity from periplasm
Kp_ratio = 1 # M, multiply by Kp to get proton binding affinity from cytoplasm
KD_ratio = 2 # M, multiply by KD to get drug binding affinity from outside
V_base = -0.15 # V, base voltage (except plot_specificity)
kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
cDc = 1e-6 # M, cytoplasmic drug concentration (except plot_efflux_vs_D)



# Axes/lists for plot_efflux_vs_KD
KD_axis = np.logspace(-9, -1.5, 200) # M, drug binding affinity
cpp_vals = np.array([1e-7, 3e-7, 6e-7, 1e-6]) # M, cytoplasmic drug concentration


#### MAIN CALLS ####

param = Params3(rD, rp, rt, cDo, cpc, vD, vp) # Create instantiation of Params3 object
Et = param.kB*param.T*np.log(KD_ratio*Kp_ratio) # Energy change associated with conformational transition (derived)

print(Et)
plot_efflux_vs_KD(param, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc, cpp_vals)