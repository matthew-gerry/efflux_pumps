'''
plot_efflux_3state.py

Plotting functions for the three-state model of bacterial efflux pumps.

Matthew Gerry, March 2023
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump


#### FUNCTIONS ####

def plot_efflux_vs_KD(param, KD_axis, Kp, V_base, kappa, cDc, cpp_vals):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY '''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        mean_output = np.vectorize(pump.efflux_MM_3)(param, KD_axis, Kp, V_base, kappa, cDc, cpp)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    KD_axis_uM = 1e6*KD_axis # KD_axis in uM
    cpp_vals_uM = [1e6*x for x in cpp_vals]
    mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1
    for i in range(len(cpp_vals)):
        plt.semilogx(KD_axis_uM, mean_efflux_nano[i],label="$[p] = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel(r"$J\:(\times 10^{-9}\:s^{-1})$")
    plt.legend()
    plt.show()


def plot_efflux_vs_D(param, KD, Kp, V_base, kappa, cDc_axis, cpp_vals):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG CONCENTRATION '''
        
    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]
        mean_output = np.vectorize(pump.efflux_MM_3)(param, KD, Kp, V_base, kappa, cDc_axis, cpp)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    cDc_axis_uM = 1e6*cDc_axis # KD_axis in uM
    cpp_vals_uM = [1e6*x for x in cpp_vals]
    mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1

    fig, ax = plt.subplots()
    for i in range(len(cpp_vals)):
        ax.plot(cDc_axis_uM, mean_efflux_nano[i],label="$[p] = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    ax.annotate("Increasing pH",xy=(20,0.002),xytext=(20,0.024),
                horizontalalignment='center', arrowprops=dict(arrowstyle='simple',lw=2))
    ax.set_xlim([0, 40])
    ax.set_ylim([0,0.062])
    ax.set_xlabel("$[D]\:(\mu M)$")
    ax.set_ylabel(r"$J\:(\times 10^{-9}\:s^{-1})$")
    ax.legend()
    plt.show()


def plot_KM(param, KD_vals, Kp, V_base, kappa, cpp_axis):
    ''' KM AS A FUNCTION OF PROTON CONCENTRATION '''

    KM_vals = [] # Initialize list of lists of KM values
    KMsimp_vals = [] # Initialize list of lists of KM values using simplified expression

    # Evaluate KM at each value of cpp
    for i in range(len(KD_vals)):
        KD = KD_vals[i]
        KM_output = np.vectorize(pump.KM_3)(param, KD, Kp, V_base, kappa, cpp_axis) # Full function
        KMsimp_output = KD*Kp*np.power(cpp_axis + Kp, -1) # Simplified expression

        KM_vals.append(KM_output)
        KMsimp_vals.append(KMsimp_output)

    fig, ax = plt.subplots()

    for i in range(len(KD_vals)):
        ax.semilogx(1e6*cpp_axis, 1e6*KM_vals[i], label="$K_D =$ "+str(int(1e6*KD_vals[i]))+" $\mu M$", linestyle=ls_list[i], linewidth=3)
    for i in range(len(KD_vals)):
        ax.semilogx(1e6*cpp_axis, 1e6*KMsimp_vals[i], '--', color="black",linewidth=1)

    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])

    ax.ticklabel_format(style='plain') # No scientific notation
    ax.set_xlabel("$[p]$ $(\mu M)$")
    ax.set_ylabel("$K_M$ $(\mu M)$")    
    plt.legend()
    plt.show()


def plot_specificity(param, KD, Kp, V_base_vals, kappa, cDc, cpp_axis):
    ''' SPECIFICITY AS A FUNCTION OF PROTON CONCENTRATION '''

    # Prepare an array of delta pH given changes in [p] to later derive a KG array
    delta_pH_axis = np.log10(param.cpc*np.power(cpp_axis,-1)) # pH difference between periplasm and cytoplasm

    S_vals = [] # Initialize list of lists of S values
    Ssimp_vals = [] # Initialize list of lists of S values using simplified expression   

    # Evaluate S at each value of the base voltage
    for i in range(len(V_base_vals)):
        V_base = V_base_vals[i]

        # Conversion between voltages and KG values
        # Derive a KG array corresponding to the different [p] values at a given V_base
        V_axis = V_base + kappa*delta_pH_axis # Based on results of Lo et al., doi: 10.1529/biophysj.106.095265
        KG_axis = np.exp(-q*V_axis/(kB*T)) # KG as a function of the proton concentrations

        # Calculation of the specificity
        S_output = np.vectorize(pump.spec_3)(param, KD, Kp, V_base, kappa, cDc, cpp_axis) # Full function
        Ssimp_output = np.multiply(cpp_axis,KG_axis) # Simplified expression

        S_vals.append(S_output)
        Ssimp_vals.append(Ssimp_output)

    KG_base_vals = [np.exp(-q*x/(kB*T)) for x in V_base_vals] # KG_base_vals to use as plot labels

    fig, ax = plt.subplots()
    for i in range(len(V_base_vals)):
        ax.semilogx(1e6*cpp_axis, S_vals[i], label="$K_G|_{\Delta\mu_p=0} =$ "+str(int(round(KG_base_vals[i]))), linestyle=ls_list[i],linewidth=3)
    for i in range(len(V_base_vals)):
        ax.semilogx(1e6*cpp_axis, Ssimp_vals[i], "--k", linewidth=1)

    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,9))

    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])

    ax.ticklabel_format(style='plain',axis='x') # No scientific notation on x axis
    ax.set_xlabel("$[p]$ $(\mu M)$")
    ax.set_ylabel("$S$")    
    plt.legend()
    plt.show()


#### GLOBAL VARIABLES ####

ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting


# Parameter values
r0 = 1 # 1/s
vD = 1 # 1/M
vp = 1 # 1/M
cDo = 1e-11 # M
cpc = 1e-7 # M

Kp = 1e-6 # M, proton binding affinity
KD = 1e-6 # M, drug binding affininty
V_base = -0.15 # V, base voltage
kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane

cDc = 1e-5 # M, cytoplasmic drug concentration

# Axes and values for computations, plotting

# For plot_efflux_vs_KD and plot_efflux_vs_D
KD_axis = np.logspace(-7, 0.5, 200) # M, drug binding affinity
cDc_axis = np.linspace(0,4e-5,100) # M, cytoplasmic drug concentration
cpp_vals = np.array([1e-7, 3e-7, 6e-7, 1e-6]) # M, cytoplasmic drug concentration

# For plot_KM and plot_specificity
KD_vals = [1e-6, 2e-6, 4e-6, 6e-6]
cpp_axis = np.logspace(-7,-5)
KG_base_vals = [40, 60, 80, 100]
V_base_vals = [-kB*T*np.log(x)/q for x in KG_base_vals]


#### MAIN CALLS ####

param = Params3(r0, cDo, cpc, vD, vp) # Create instantiation of Params3 object

plot_efflux_vs_KD(param, KD_axis, Kp, V_base, kappa, cDc, cpp_vals)
plot_efflux_vs_D(param, KD, Kp, V_base, kappa, cDc_axis, cpp_vals)
plot_KM(param, KD_vals, Kp, V_base, kappa, cpp_axis)
plot_specificity(param, KD, Kp, V_base_vals, kappa, cDc, cpp_axis)