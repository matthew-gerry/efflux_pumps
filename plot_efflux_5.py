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


def plot_efflux_vs_ratio(param, KD, Kp, KD_ratio_axis, Kp_ratio, V_base, kappa, cDc, cpp_vals):
    ''' PLOT THE MEAN EFFLUX AS A FUNCTION OF THE RATIOS BETWEEN DRUG BINDING AFFINITIES '''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        mean_output = np.vectorize(pump.efflux_numerical_5)(param, KD, Kp, KD_ratio_axis*KD, Kp_ratio*Kp, V_base, kappa, cDc, cpp)

        mean_efflux.append(mean_output)

    # Calculate the energy associated with the conformational transition
    Et_eV = 6.24e18*param.kB*param.T*np.log(Kp_ratio*KD_ratio_axis) # eV

    # Plot mean values and variances side by side
    cpp_vals_uM = [1e6*x for x in cpp_vals]

    fig, ax1 = plt.subplots()
    for i in range(len(cpp_vals)):
        ax1.semilogx(KD_ratio_axis, mean_efflux[i], label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    ax1.set_xlabel("$K_{D,out}/K_{D,in}$")
    ax1.set_ylabel("$J\:(s^{-1})$")
    ax1.legend()
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)

    ax2 = fig.add_axes([0.21,0.37,0.2,0.2])
    ax2.semilogx(KD_ratio_axis, Et_eV, '-k')
    ax2.text(0.008,0.1,"$E_t\:(eV)$")

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


# Variables - all functions
Kp = 1e-6 # M, proton binding affinity from periplasm
Kp_ratio = 1 # M, multiply by Kp to get proton binding affinity from cytoplasm
V_base = -0.15 # V, base voltage
kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
cDc = 1e-6 # M, cytoplasmic drug concentration


# Variables for plot_efflux_vs_KD
KD_axis = np.logspace(-8.5, -2.5, 200) # M, drug binding affinity
cpp_vals = np.array([1e-7, 3e-7, 6e-7, 1e-6]) # M, cytoplasmic drug concentration
KD_ratio = 2 # M, multiply by KD to get drug binding affinity from outside

# Variables for plot_efflux_vs_ratio
KD = 1e-6 # M, drug binding affinity from inside
KD_ratio_axis = np.logspace(-2,3,50) # Ratio of outside to inside drug binding affinities


#### MAIN CALLS ####

param = Params3(rD, rp, rt, cDo, cpc, vD, vp) # Create instantiation of Params3 object


# plot_efflux_vs_KD(param, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc, cpp_vals)
plot_efflux_vs_ratio(param, KD, Kp, KD_ratio_axis, Kp_ratio, V_base, kappa, cDc, cpp_vals)