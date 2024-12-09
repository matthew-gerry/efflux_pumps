'''
plot_efflux_5.py

Plotting functions for the five-state model of bacterial efflux pumps.

When calling functions, an object of the class Parameters should be passed to as 
argument denoted param (three-state and five-state models share a Parameters class).

Matthew Gerry, June 2023
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump

plt.rcParams['figure.dpi'] = 300 # Improve default resolution of figures

#### FUNCTIONS ####

def plot_efflux_vs_KD(param, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc, cpp_vals, reversed_unbinding=False):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY '''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        mean_output = []
        
        mean_output = np.vectorize(pump.efflux_numerical_5)(param, KD_axis, Kp, KD_ratio*KD_axis, Kp_ratio*Kp, V_base, kappa, cDc, cpp, reversed_unbinding)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    KD_axis_uM = 1e6*KD_axis # KD_axis in uM
    cpp_vals_uM = [1e6*x for x in cpp_vals]
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1
    for i in range(len(cpp_vals)):
        plt.semilogx(KD_axis_uM, mean_efflux[i]/param.rD, label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    
    plt.xlim([min(KD_axis_uM), max(KD_axis_uM)])
    plt.ylim([0, 8.1e-8])
    
    plt.xlabel("$K_D\:(\mu M)$", fontsize=14)
    plt.ylabel(r"$J\nu_D/k_D^+$", fontsize=14)
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    plt.tick_params(labelsize=12)

    plt.text(3e-2,7e-8,"A",fontsize=18)
    plt.legend(fontsize=14)
    plt.show()
    # plt.savefig("efflux_vs_KD_5.png")

#### GLOBAL VARIABLES ####

ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting


# Parameter values
rD = 1e8 # 1/s
rp = 1e14 # 1/s
rt = 1e6 # 1/s, unlike in the three-state model, rt does not depend on other char. rates
vD = 1 # 1/M
vp = 1e-6 # 1/M
cDo = 1e-5 # M
cpc = 1e-7 # M


# Variables - multiple functions
Kp = 1e-7 # M, proton binding affinity from periplasm
Kp_ratio = 1 # M, multiply by Kp to get proton binding affinity from cytoplasm
V_base = -np.log(100)*kB*T/q # V, base voltage, about -110 mV
kappa = 0 # If desired, a coefficient governing how the membrane potential varies linearly with pH difference - set to zero for our study
cDc = 1e-5 # M, cytoplasmic drug concentration

KD_ratio = 10 # M, multiply by KD to get drug binding affinity from outside

# Variables for plot_efflux_vs_KD
KD_axis = np.logspace(-9, -1, 200) # M, drug binding affinity
cpp_vals = np.array([1e-7, 3e-7, 6e-7, 1e-6]) # M, cytoplasmic drug concentration


#### MAIN CALLS ####

param = Parameters(rD, rp, rt, cDo, cpc, vD, vp) # Create instantiation of Parameters object
plot_efflux_vs_KD(param, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc, cpp_vals, reversed_unbinding=True)