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

def plot_efflux_vs_KD(param, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc, cpp_vals, reversed_unbinding=False):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY '''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        mean_output = []
        
        mean_output = np.vectorize(pump.efflux_numerical_5)(param, KD_axis, Kp, KD_ratio*KD_axis, Kp_ratio*Kp, V_base, kappa, cDc, cpp, reversed_unbinding)
        # for j in range(len(KD_axis)):
        #     KD = KD_axis[j]

        #     output_val = pump.efflux_numerical_5(param, KD, Kp, KD_ratio*KD, Kp_ratio*Kp, V_base, kappa, cDc, cpp, reversed_unbinding)
        #     mean_output.append(output_val)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    KD_axis_uM = 1e6*KD_axis # KD_axis in uM
    cpp_vals_uM = [1e6*x for x in cpp_vals]
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1
    for i in range(len(cpp_vals)):
        plt.semilogx(KD_axis_uM, mean_efflux[i]/param.rD, label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    
    plt.xlim([min(KD_axis_uM), max(KD_axis_uM)])
    # plt.ylim([0, 7e-6])
    
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel(r"$J\nu_D/k_D^+$")
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    
    plt.text(5e-3,5.8e-6,"A",fontsize=18)
    plt.legend()
    plt.show()


def plot_efflux_vs_ratio(param, KD, Kp, KD_ratio_axis, Kp_ratio, V_base, kappa, cDc, cpp_vals, reversed_unbinding=False):
    ''' PLOT THE MEAN EFFLUX AS A FUNCTION OF THE RATIOS BETWEEN DRUG BINDING AFFINITIES '''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        mean_output = np.vectorize(pump.efflux_numerical_5)(param, KD, Kp, KD_ratio_axis*KD, Kp_ratio*Kp, V_base, kappa, cDc, cpp, reversed_unbinding)

        mean_efflux.append(mean_output)

    # Calculate the energy associated with the conformational transition
    Et_eV = 6.24e18*param.kB*param.T*np.log(Kp_ratio*KD_ratio_axis) # eV

    # Plot mean values and variances side by side
    cpp_vals_uM = [1e6*x for x in cpp_vals]

    fig, ax1 = plt.subplots()
    for i in range(len(cpp_vals)):
        ax1.semilogx(KD_ratio_axis, mean_efflux[i], label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    ax1.set_ylim([0, 5])
    ax1.set_xlabel(r"$\tilde{K_D}/K_D$")
    ax1.set_ylabel("$J\:(s^{-1})$")
    ax1.legend()
    ax1.text(7, 4.5, "(B)")
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)

    ax2 = fig.add_axes([0.215,0.57,0.19,0.2])

    ax2.semilogx(KD_ratio_axis, Et_eV, '-k')
    ax2.text(0.008,0.16,"$E_t\:(eV)$")

    # ax2.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    # ax2.ticklabel_format(style='plain',axis='x') # No scientific notation on x axis
    ax2.set_xticks([0.01, 1, 100]) 
    
    plt.show()

def plot_efflux_vs_D_2(param, KD_vals, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc_axis, cpp, reversed_unbinding=False):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG CONCENTRATION, VARYING KD
        EQUAL BINDING AFFINITY FROM INSIDE AND OUT FOR SIMPLICITY '''


    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(KD_vals)):
        KD = KD_vals[i]
        mean_output = np.vectorize(pump.efflux_numerical_5)(param, KD, Kp, KD*KD_ratio, Kp*Kp_ratio, V_base, kappa, cDc_axis, cpp, reversed_unbinding)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    cDc_axis_uM = 1e6*cDc_axis # KD_axis in uM
    KD_vals_uM = [1e6*x for x in KD_vals]
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1

    fig, ax = plt.subplots()
    for i in range(len(KD_vals)):
        ax.semilogy(cDc_axis_uM, mean_efflux[i],label="$K_D = "+str(int(KD_vals_uM[i]))+"\:\mu M$", linestyle = ls_list[i])
    ax.annotate("Weaker binding",xy=(17.5,5e-3),xytext=(17.5,0.04),
                horizontalalignment='center', arrowprops=dict(arrowstyle='simple',lw=2))
    # ax.set_xlim([0, max(cDc_axis_uM)])
    ax.set_xlabel("$[D]_{in}\:(\mu M)$")
    ax.set_ylabel("$J\:(s^{-1})$")
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    ax.set_yticks([5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1,5e-1, 1])
    # ax.set_ylim([0.002,1.8])
    ax.text(17, 1.2, "(A)")
    ax.legend()
    plt.show()


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
# kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
kappa = 0
cDc = 1e-5 # M, cytoplasmic drug concentration

KD_ratio = 10 # M, multiply by KD to get drug binding affinity from outside

# Variables for plot_efflux_vs_KD
KD_axis = np.logspace(-9, -1, 200) # M, drug binding affinity
cpp_vals = np.array([1e-7, 3e-7, 6e-7, 1e-6]) # M, cytoplasmic drug concentration

# Variables for plot_efflux_vs_ratio
KD = 1e-6 # M, drug binding affinity from inside
KD_ratio_axis = np.logspace(-2.5,4.5,50) # Ratio of outside to inside drug binding affinities

# Variable for plot_efflux_vs_KD_2
cDc_axis = np.linspace(0,3.5e-5,100)
KD_vals = [1e-6, 5e-6, 1e-5, 1e-4]
cpp = 1e-6

#### MAIN CALLS ####

param = Params3(rD, rp, rt, cDo, cpc, vD, vp) # Create instantiation of Params3 object


plot_efflux_vs_KD(param, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc, cpp_vals, reversed_unbinding=True)
# plot_efflux_vs_ratio(param, KD, Kp, KD_ratio_axis, Kp_ratio, V_base, kappa, cDc, cpp_vals, reversed_unbinding=True)
# plot_efflux_vs_D_2(param, KD_vals, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc_axis, cpp, reversed_unbinding=False)