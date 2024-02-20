'''
plot_efflux_3.py

Plotting functions for the three-state model of bacterial efflux pumps.

When calling functions, an object of the class Params3 should be passed to as 
argument denoted param.

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
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1
    for i in range(len(cpp_vals)):
        plt.semilogx(KD_axis_uM, mean_efflux[i]/param.rD, label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    # plt.ylim(0,9e-6)
    plt.xlim(min(KD_axis_uM),max(KD_axis_uM))
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel(r"$J\nu_D/k_D^+$")
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    plt.legend()
    plt.text(10**(-2.5),7.7e-6,"A",fontsize=18)
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
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1

    fig, ax = plt.subplots()
    for i in range(len(cpp_vals)):
        ax.plot(cDc_axis_uM, mean_efflux[i],label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    ax.annotate("Increasing pH",xy=(25,0.000025),xytext=(25,0.00027),
                horizontalalignment='center', arrowprops=dict(arrowstyle='simple',lw=2))
    ax.set_xlim([0, max(cDc_axis_uM)])
    ax.set_ylim([0,6.5e-4])
    ax.set_xlabel("$[D]_{in}\:(\mu M)$")
    ax.set_ylabel("$J\:(s^{-1})$")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    ax.legend()
    plt.show()


def plot_efflux_vs_D_2(param, KD_vals, Kp, V_base, kappa, cDc_axis, cpp):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG CONCENTRATION, VARYING KD '''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(KD_vals)):
        KD = KD_vals[i]
        mean_output = np.vectorize(pump.efflux_MM_3)(param, KD, Kp, V_base, kappa, cDc_axis, cpp)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    cDc_axis_uM = 1e6*cDc_axis # KD_axis in uM
    KD_vals_uM = [1e6*x for x in KD_vals]
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1

    fig, ax = plt.subplots()
    for i in range(len(KD_vals)):
        ax.semilogy(cDc_axis_uM, mean_efflux[i]/param.rD,label="$K_D = "+str(int(KD_vals_uM[i]))+"\:\mu M$", linestyle = ls_list[i])
    ax.set_xlabel("$[D]_{in}\:(\mu M)$")
    ax.set_ylabel(r"$J\nu_D/k_D^+$")
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    # ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    # ax.set_yticks([1e-7,2e-7,5e-7,1e-6,2e-6,5e-6,1e-5])
    ax.set_xlim([0,40])
    ax.annotate("Stronger binding",xy=(20,1.5e-5), xytext=(20, 1.5e-6),
            horizontalalignment='center', arrowprops=dict(arrowstyle='simple',lw=2))    
    ax.text(2.5,8e-6,"A",fontsize=16)
    ax.legend()
    plt.show()


def plot_KM(param, KD_vals, Kp, V_base, kappa, cpp_axis):
    ''' KM AS A FUNCTION OF PROTON CONCENTRATION '''

    KM_vals = [] # Initialize list of lists of KM values
    KMsimp_vals = [] # Initialize list of lists of KM values using simplified expression

    KG = np.exp(-param.q*V_base/(param.kB*param.T)) # KG at V_base (exact if kappa=0)

    # Evaluate KM at each value of cpp
    for i in range(len(KD_vals)):
        KD = KD_vals[i]
        KM_output = np.vectorize(pump.KM_3)(param, KD, Kp, V_base, kappa, cpp_axis)[0] # Full function
        KMsimp_output = np.vectorize(pump.KM_3)(param, KD, Kp, V_base, kappa, cpp_axis)[1] # Approximate expression

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

    # ax.set_xlim([0.1,10])
    # ax.set_ylim([0,1000])

    ax.ticklabel_format(style='plain') # No scientific notation
    ax.set_xlabel("$[p]_{per}$ $(\mu M)$")
    ax.set_ylabel("$K_M$ $(\mu M)$")
    ax.text(0.13,850,"B",fontsize=16)
    # plt.legend()
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
        Ssimp_output = param.rt*param.vp*np.multiply(cpp_axis,KG_axis) # Simplified expression

        S_vals.append(S_output)
        Ssimp_vals.append(Ssimp_output)

    KG_base_vals = [np.exp(-q*x/(kB*T)) for x in V_base_vals] # KG_base_vals to use as plot labels

    S_micro = [1e-6*x for x in S_vals]
    Ssimp_micro = [1e-6*x for x in Ssimp_vals]

    fig, ax = plt.subplots()
    for i in range(len(V_base_vals)):
        ax.semilogx(1e6*cpp_axis, S_micro[i], label="$K_G|_{\Delta\mu_p=0} =$ "+str(int(round(KG_base_vals[i]))), linestyle=ls_list[i],linewidth=3)
    for i in range(len(V_base_vals)):
        ax.semilogx(1e6*cpp_axis, Ssimp_micro[i], "--k", linewidth=1)

    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,9))

    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])

    ax.ticklabel_format(style='plain',axis='x') # No scientific notation on x axis
    ax.set_xlabel("$[p]_{per}$ $(\mu M)$")
    ax.set_ylabel("$S$ $(\mu M^{-1}s^{-1})$")
    ax.text(0.9,1.05e-3,'(B)',fontsize='large')
    plt.legend()
    plt.show()


def plot_efflux_vs_D_over_KD(param, KD_vals, Kp, V_base, kappa, cDc_over_KD_axis, cpp):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG CONCENTRATION OVER KD, VARYING KD '''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(KD_vals)):
        KD = KD_vals[i]
        cDc_axis = KD*cDc_over_KD_axis

        mean_output = np.vectorize(pump.efflux_MM_3)(param, KD, Kp, V_base, kappa, cDc_axis, cpp)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    KD_vals_uM = [1e6*x for x in KD_vals]
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1

    fig, ax = plt.subplots()
    for i in range(len(KD_vals)):
        ax.semilogx(cDc_over_KD_axis, mean_efflux[i],label="$K_D = "+str(int(KD_vals_uM[i]))+"\:\mu M$", linestyle = ls_list[i])
    # ax.annotate("Stronger binding",xy=(25,3e-4),xytext=(25,1.5e-3),
                # horizontalalignment='center', arrowprops=dict(arrowstyle='simple',lw=2))
    ax.set_xlim([min(cDc_over_KD_axis), max(cDc_over_KD_axis)])
    ax.set_ylim([0,1.1])
    ax.set_xlabel("$[D]_{in}/K_D$")
    ax.set_ylabel("$J\:(s^{-1})$")
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    # ax.set_yticks([5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    ax.legend()
    plt.show()


def contour_efflux_p_V(param, KD, Kp, V_abs_axis, kappa, cDc, cpp_axis, filename):
    ''' CONTOUR PLOT OF THE EFFLUX AS A FUNCTION OF BOTH [p] AND THE MAGNITUDE OF V '''
   
    # Note data is saved in/loaded from the parent directory
    try: # Load data if saved
        J_vals = np.load("../"+filename+".npy")
        '''
        IF LOADING DATA, ENSURE THAT V_axis AND cpp_axis FED INTO THIS FUNCTION
        MATCH THOSE USED TO CALCULATE DATA
        '''
    except: # Otherwise calculate and save the efflux values
        J_vals = np.zeros([len(V_abs_axis), len(cpp_axis)])

        for i in range(len(V_abs_axis)):
            V_base = -V_abs_axis[i]

            J_vals[i,:] = np.vectorize(pump.efflux_MM_3)(param, KD, Kp, V_base, kappa, cDc, cpp_axis)
        np.save("../"+filename+".npy", J_vals) # Save data for next time (good if just playing around with plot formatting, bad if changing param values on consecutive runs)

    fig, ax = plt.subplots()

    dmu_p_axis = 6.24e21*kB*T*np.log(cpp_axis/param.cpc) # Will plot against the proton chem potential difference (in meV), not the concentration
    V_abs_milli = 1e3*V_abs_axis
    [X, Y] = np.meshgrid(dmu_p_axis, V_abs_milli)

    sctr = ax.scatter(X,Y,c=J_vals,marker='x')
    cbar = fig.colorbar(sctr)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel("$\Delta\mu_{protons}\;(meV)$")
    ax.set_ylabel("$|V|\;(mV)$")
    cbar.ax.set_ylabel("$J\:(s^{-1})$")

    plt.show()


def linear_response_check(param_list, KD, Kp, V_base, kappa, dmuD, dmup_axis):
    ''' PLOT EFFLUX AS A FUNCTION OF PROTON CHEMICAL POTENTIAL DIFFERENCE AT LINEAR RESPONSE '''

    # Notice that the quantity dmup is the chemical potential difference for protons due ONLY to the concentration gradient (voltage is treated separately)
    # and that it is defined as mu_cyt - mu_per, so it will in general be negative
    #   We will ultimately plot the efflux against the negative of this quantity (more intuitive)

    # Empty lists to populate with the results of calculations
    exact_efflux = []
    lr_efflux = []

    for i in range(len(param_list)):
        param = param_list[i]

        # Evaluate pre-factor at equilibrium
        cpp_axis = param.cpc*np.exp(-dmup_axis/(kB*T)) # Get periplasmic [p] given the chem potential difference
        KM_lr = pump.KM_3(param, KD, Kp, 0, 0, param.cpc)[0] # Get equilibirum KM value
        ktplus_lr = param.rt*param.vD*param.vp*KD*Kp/(1 + param.vD*param.vp*KD*Kp) # Get ktplus assuming zero voltage

        # Calculate the inner drug concentration given the outer and dmuD
        cDc = param.cDo*np.exp(-dmuD/(kB*T))

        # Calculate the exact value of the efflux
        exact_output = np.vectorize(pump.efflux_MM_3)(param, KD, Kp, V_base, kappa, cDc, cpp_axis)

        # Calulate the thermodynamic force based on the exact values of the concentrations and voltage
        force = -(q*V_base + dmuD + dmup_axis)/(kB*T)
        # Combine with equiliibrum values calculated above to get the linear response efflux
        lr_output = ktplus_lr*(param.cpc/(Kp+ param.cpc))*(param.cDo/(KM_lr+param.cDo))*force

        # Record flipped version of these lists since we are plotting against -dmup (see comments above)
        exact_efflux.append(np.flip(exact_output))
        lr_efflux.append(np.flip(lr_output))

    # Plot mean values and variances side by side
    dmup_axis_plot = -np.flip(dmup_axis)/(kB*T)

    fig, ax = plt.subplots()
    for i in range(len(param_list)):
        param = param_list[i]
        ax.plot(dmup_axis_plot, lr_efflux[i]/param.rD, label=r"$\bar{[D]} = "+str(round(1e6*param.cDo,1))+"\:\mu M$", linestyle = ls_list[i])
        ax.plot(dmup_axis_plot, exact_efflux[i]/param.rD, '--', color="black", linewidth=1)
    ax.set_xlim([0, max(dmup_axis_plot)])
    # ax.set_ylim([0,6.5e-4])
    ax.set_xlabel("$\ln([p]_{per}/[p]_{cyt})$")
    ax.set_ylabel(r"$J\nu_D/k_D^+$")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    ax.legend()
    plt.show()


#### GLOBAL VARIABLES ####

ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting


# Parameter values
rD = 1e8 # 1/s
rp = 1e14 # 1/s
rt = 1e18 # 1/s
vD = 1 # 1/M
vp = 1e-6 # 1/M
cDo = 1e-5 # M
cpc = 1e-7 # M

# Variables - for all functions
Kp = 1e-7 # M, proton binding affinity
# V_base = -0.15 # V, base voltage (except plot_specificity)
V_base = -np.log(100)*kB*T/q  # V, base voltage, about -110 mV (except plot_specificity)
# kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
kappa = 0
KD = 1e-5 # M, drug binding affininty (except plot_efflux_vs_KD)
cDc = 1e-5 # M, cytoplasmic drug concentration (except plot_efflux_vs_D)

# For plot_efflux_vs_KD and plot_efflux_vs_D
KD_axis = np.logspace(-9, -1, 200) # M, drug binding affinity
cDc_axis = np.linspace(0,5e-5,100) # M, cytoplasmic drug concentration
cpp_vals = np.array([1e-7, 3e-7, 6e-7, 1e-6]) # M, cytoplasmic drug concentration

# For plot_KM and plot_specificity
# KD_vals = [1e-6, 2e-6, 4e-6, 6e-6]
cpp_axis = np.logspace(-7,-5, 200)
KG_base_vals = [40, 60, 80, 100]
V_base_vals = [-kB*T*np.log(x)/q for x in KG_base_vals]

# For plot_efflux_vs_D_2
cDc_axis_2 = np.linspace(0,4e-5,100)
KD_vals_2 = [5e-6, 1e-5, 4e-5, 7.5e-5]
cpp = 3e-7

# For plot_efflux_vs_D_over_KD
cDc_over_KD_axis = np.logspace(-2.2,3.5,100)

# For contour_efflux_p_V
V_abs_axis = np.linspace(0.001, 0.15, 225)
cpp_axis_2 = np.logspace(-7, -5, 200)

# For linear_response_check
# V_base_vals_lr = [-0.000, -0.0001, -0.0005] # V, membrane potential
V_base_lr = 0
dmuD = 0 # Assume no drug concentration gradient
dmup_axis_lr = np.linspace(-3e-22,0,100) # J, proton concentration gradient chemical potential
cDoA = 1e-6 # Some alternate values of the drug concentrations
cDoB = 1e-7

#### MAIN CALLS ####

param = Params3(rD, rp, rt, cDo, cpc, vD, vp) # Create instantiation of Params3 object
paramA = Params3(rD, rp, rt, cDoA, cpc, vD, vp)
paramB = Params3(rD, rp, rt, cDoB, cpc, vD, vp)
param_list = [param, paramA, paramB]

plot_efflux_vs_KD(param, KD_axis, Kp, V_base, kappa, cDc, cpp_vals)
# plot_efflux_vs_D(param, KD, Kp, V_base, kappa, cDc_axis, cpp_vals)
plot_KM(param, KD_vals_2, Kp, V_base, kappa, cpp_axis)
# plot_specificity(param, KD, Kp, V_base_vals, kappa, cDc, cpp_axis)
plot_efflux_vs_D_2(param, KD_vals_2, Kp, V_base, kappa, cDc_axis_2, cpp)
# plot_efflux_vs_D_over_KD(param, KD_vals_2, Kp, V_base, kappa, cDc_over_KD_axis, cpp)
# contour_efflux_p_V(param, KD, Kp, V_abs_axis, kappa, cDc, cpp_axis_2, "efflux_p_V_data")
# linear_response_check(param_list, KD, Kp, V_base_lr, kappa, dmuD, dmup_axis_lr)