'''
plot_efflux_8state.py

Plotting functions for the three-state model of bacterial efflux pumps.

This includes functions to numerically estimate KM and, in turn, the specificity,
since we do not have analytic expressions for these quantities as we do for the
three-state model.

Matthew Gerry, March 2023
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump


#### FUNCTIONS ####

# Get values in units 9 orders of magnitude smaller
nanofy = lambda a : [1e9*x for x in a]

def plot_efflux_vs_KD(param, KD_axis, KDA, Kp_list, V_base, kappa, cDc, cpp_vals):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY '''

    efflux_vals = len(cpp_vals)*[[]] # List to populate with values of the mean efflux

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        for j in range(len(KD_axis)):
            KD_list = [KDA, KD_axis[j]]
        
            efflux_output = pump.efflux_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)

            efflux_vals[i] = efflux_vals[i] + [efflux_output]
        
    
    # Plot mean values and variances side by side
    KD_axis_uM = 1e6*KD_axis # KD_axis in uM
    cpp_vals_uM = [1e6*x for x in cpp_vals]
    mean_efflux_nano = [nanofy(y) for y in efflux_vals] 
    for i in range(len(cpp_vals)):
        plt.semilogx(KD_axis_uM, mean_efflux_nano[i],label="$[p] = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel(r"$J\:(\times 10^{-9}\:s^{-1})$")
    plt.legend()
    plt.show()


#### GLOBAL VARIABLES ####

ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting

# Parameter values
r0 = 1 # 1/s
vD_list = [1,1] # 1/M
vp_list = [1,1,1,1] # 1/M
cDo = 1e-11 # M
cpc = 1e-7 # M

Kp_list = [1e-6, 1e-6, 1e-6, 1e-6] #M, proton binding affinities
KDA = 1e-6 # M, drug binding affinity for A cycle
V_base = -0.15 # V, base voltage (except plot_specificity)
kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
cDc = 1e-5 # M, cytoplasmic drug concentration (except plot_efflux_vs_D)

# For plot_efflux_vs_KD
cpp_vals = [1e-7, 3e-7, 6e-7, 1e-6]
KD_axis = np.logspace(-5,-1, 100)


#### MAIN CALLS ####

param = Params8(r0, cDo, cpc, vD_list, vp_list)

plot_efflux_vs_KD(param, KD_axis, KDA, Kp_list, V_base, kappa, cDc, cpp_vals)