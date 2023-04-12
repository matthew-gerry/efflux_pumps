'''
plot_efflux_8state.py

Plotting functions for the three-state model of bacterial efflux pumps.

This includes functions to numerically estimate KM and, in turn, the specificity,
since we do not have analytic expressions for these quantities as we do for the
three-state model. 

When calling functions, an object of the class Params8 should be passed to as 
argument denoted param.

NOTE: the functions in this file assume that when the drug binding affinity is varied,
it is only that on cycle B. The drug binding affinity on cycle A is held fixed and passed
to many functions as the argument KDA. If one wishes to vary both KD values, or instead 
constrain the system such that the two values are always equal to one another, this file
must be edited accordingly.

Matthew Gerry, March 2023
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump


#### FUNCTIONS: CALCULATIONS ####

def get_efflux_vs_D(param, KD_list, Kp_list, V_base, kappa, cDc_axis, cpp_vals):
    ''' GET A LIST OF LISTS HOLDING EFFLUX VS DRUG CONCENTRATION VALUES FOR VARIOUS PROTON CONCENTRATIONS '''

    efflux_vals = []

    # Evaluate mean efflux at each value of cpp and [D]
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        efflux_at_fixed_cpp = []
        for j in range(len(cDc_axis)):
            cDc = cDc_axis[j]

            efflux_output = pump.efflux_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)
            efflux_at_fixed_cpp.append(efflux_output)

        efflux_vals.append(efflux_at_fixed_cpp)
    
    return efflux_vals


def get_KM(param, KD_list, Kp_list, V_base, kappa, cDc_axis, cpp_axis):
    ''' USE NUMERICALLY CALCULATED EFFLUX VS [D] TO APPROXIMATE KM ([D] AT HALF MAX EFFLUX) '''

    KM_vals = np.zeros(len(cpp_axis)) # Allocate array for KM as a function of [p]

    efflux_matrix = get_efflux_vs_D(param, KD_list, Kp_list, V_base, kappa, cDc_axis, cpp_axis)

    for j in range(len(cpp_axis)): # Run down the list of cpp values, approximate KM for each
        efflux_curve = efflux_matrix[j]
        Jmax = efflux_curve[-1] # Efflux should grow monotonically with [D]
        # Roughly confirm that we reach high enough [D] to approach the max efflux
        if Jmax-efflux_curve[int(len(efflux_curve)/2)]<0.9*Jmax:
            # Identify the position in the list of the next efflux value after half max is achieved
            half_max_index = next(index for index, val in enumerate(efflux_curve) if val >= Jmax/2)
            KM = cDc_axis[half_max_index] # KM is appproximated as the cytoplasmic drug concentration at this point
            KM_vals[j] = KM
        else:
            print("Efflux curve is not approaching its max value. Probe higher [D] values.")
            break

    return KM_vals


#### FUNCTIONS: PLOTTING ####

# Get values in smaller units
nanofy = lambda a : [1e9*x for x in a] # By 9 orders of magnitude
microfy = lambda a : [1e6*x for x in a] # By 6 orders of magnitude

def plot_efflux_vs_KD(param, KD_axis, Kp_list, V_base, kappa, cDc, cpp_vals):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY '''

    efflux_vals = len(cpp_vals)*[[]] # List to populate with values of the mean efflux

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        for j in range(len(KD_axis)):
            KD_list = 2*[KD_axis[j]]
        
            efflux_output = pump.efflux_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)

            efflux_vals[i] = efflux_vals[i] + [efflux_output]
        
    # Configure some things for plotting
    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting
    KD_axis_uM = microfy(KD_axis) # KD_axis in uM
    cpp_vals_uM = microfy(cpp_vals)

    for i in range(len(cpp_vals)):
        plt.semilogx(KD_axis_uM, efflux_vals[i],label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel("$J\:(s^{-1})$")
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    plt.legend()
    plt.show()


def plot_efflux_vs_D(param, KD_list, Kp_list, V_base, kappa, cDc_axis, cpp_vals):
    ''' PLOT EFFLUX AS A FUNCTION OF DRUG CONCENTRATION AT VARIOUS VALUES OF PROTON CONCENTRATION '''

    efflux_vals = get_efflux_vs_D(param, KD_list, Kp_list, V_base, kappa, cDc_axis, cpp_vals)

    # Configure some things for plotting
    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting
    cDc_axis_uM = microfy(cDc_axis) # KD_axis in uM - this works because cDc_axis is a numpy array
    cpp_vals_uM = microfy(cpp_vals)

    fig, ax = plt.subplots()
    for i in range(len(cpp_vals)):
        ax.plot(cDc_axis_uM, efflux_vals[i],label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    
    ax.annotate("Increasing pH",xy=(0.6,2e-6),xytext=(0.6,2.3e-5),
                horizontalalignment='center', arrowprops=dict(arrowstyle='simple',lw=2))
    ax.set_xlim([0, 1.2])
    ax.set_ylim([0,6.3e-5])
    ax.set_xlabel("$[D]_{cyt}\:(\mu M)$")
    ax.set_ylabel("$J\:(s^{-1})$")
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    ax.legend()
    plt.show()


def plot_KM(param, KD_vals, Kp_list, V_base, kappa, cDc_axis, cpp_axis):
    ''' PLOT NUMERICALLY APPROXIMATED KM VALUES AS A FUNCTION OF [p] '''

    KM_vals = []

    for i in range(len(KD_vals)): # Get KM as a function of [p] at each KD (B) value
        KD_list = 2*[KD_vals[i]]

        KM_vals.append(get_KM(param, KD_list, Kp_list, V_base, kappa, cDc_axis, cpp_axis))

    # Configure some things for plotting
    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting
    cpp_axis_uM = microfy(cpp_axis)
    KM_vals_uM = [microfy(y) for y in KM_vals]

    for i in range(len(KD_vals)):
        plt.semilogx(cpp_axis_uM, KM_vals_uM[i], label="$K_D =$ "+str(round(1e6*KD_vals[i]))+" $ \mu M$", linestyle = ls_list[i])
    plt.xlabel("$[p]_{per}\:(\mu M)$")
    plt.ylabel("$K_M\:(\mu M)$")
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    plt.legend()
    plt.show()


def plot_efficiency_vs_p(param, KD_vals, Kp_list, V_base, kappa, cDc, cpp_axis):
    ''' PLOT CHEMICAL EFFICIENCY AS A FUNCTION OF [p] FOR VARIOUS KD VALUES '''

    efficiency = len(KD_vals)*[[]] # Allocate array for chemical efficiency values

    # Evaluate efflux at each value of periplasmic proton concentration
    for i in range(len(KD_vals)):
        KD_list = 2*[KD_vals[i]] # Cycle through different values of K_D (cycle B)

        for j in range(len(cpp_axis)):
            cpp = cpp_axis[j]
            
            efflux = pump.efflux_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)
            p_flux = pump.p_flux_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)
            efficiency[i] = efficiency[i] + [efflux/p_flux] # Chemical efficiency

    # Configure some things for plotting
    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting
    KD_vals_uM = microfy(KD_vals) # KD_axis in uM
    cpp_axis_uM = microfy(cpp_axis) # cpp_vals in uM

    plt.figure()
    for i in range(len(KD_vals)):
        plt.semilogx(cpp_axis_uM, efficiency[i],label="$K_D =\:"+str(int(KD_vals_uM[i]))+"\:\mu M$",linestyle=ls_list[i])
        plt.xlabel("$[p]_{per}\:(\mu M)$")
        plt.ylabel("$J/J_p$")
    plt.legend()
    plt.show()
    

def plot_compare_fluxes(param, KD_axis, Kp_list, V_base, kappa, cDc, cpp_vals):
    ''' COMPARE THE PROTON FLUX AND DRUG EFFLUX WITH THE MULTICYCLIC MODEL OVER A RANGE OF K_D '''

    efflux, p_flux, efficiency = len(cpp_vals)*[[]], len(cpp_vals)*[[]], len(cpp_vals)*[[]]

    # Evaluate efflux mean and variance at each value of periplasmic proton concentration
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i] # Cycle through different values of [p]

        for j in range(len(KD_axis)):
            KD_list = 2*[KD_axis[j]]

            efflux_output = pump.efflux_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)
            p_flux_output = pump.p_flux_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)
            
            efflux[i] = efflux[i] + [efflux_output] # Drug efflux
            p_flux[i] = p_flux[i] + [p_flux_output] # Proton flux

        efficiency[i] = [a/b for a,b, in zip(efflux[i],p_flux[i])] # Chemical efficiency

        # Configure some things for plotting
    KD_axis_uM = microfy(KD_axis) # KD_axis in uM
    cpp_vals_uM = microfy(cpp_vals)
    # efflux_nano = [nanofy(y) for y in efflux]
    # p_flux_nano = [nanofy(y) for y in p_flux]

    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyles
    colour_list = ["tab:blue","tab:orange","tab:green","tab:red"] # Plot colours

    plt.subplot(1,2,1)
    for i in range(len(cpp_vals)):
        plt.loglog(KD_axis_uM, efflux[i],label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", color=colour_list[i], linestyle=ls_list[0])
        plt.loglog(KD_axis_uM, p_flux[i], color=colour_list[i], linestyle=ls_list[2])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel("$J\:(s^{-1})$")
    # plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    # plt.legend()

    plt.subplot(1,2,2)
    for i in range(len(cpp_vals)):
        plt.loglog(KD_axis_uM, efficiency[i],label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", color=colour_list[i])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel("$J/J_p$")
    plt.legend()

    plt.show()


#### GLOBAL VARIABLES ####


# Parameter values
rD = 1e6 # 1/s
rp = 1e6 # 1/s
rt = 1e6 # 1/s
vD_list = [1,1] # 1/M
vp_list = [0.1,0.1,0.1,0.1] # 1/M
cDo = 1e-11 # M
cpc = 1e-7 # M

Kp_list = [1e-6, 1e-6, 1e-6, 1e-6] #M, proton binding affinities
V_base = -0.15 # V, base voltage (except plot_specificity)
kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
cDc = 1e-5 # M, cytoplasmic drug concentration (except plot_efflux_vs_D)

# For plot_efflux_vs_KD
cpp_vals = [1e-7, 3e-7, 6e-7, 1e-6]
KD_axis = np.logspace(-5.5, -1.5, 100) # Also for plot_compare_fluxes

# For plot_efflux_vs_D
KD_list = [1e-5, 1e-5]
cDc_axis = np.linspace(cDo,1.2e-6,1200) # Also for plot_KM (need high resolution)

# For plot_KM
KD_vals = [1e-6, 2e-6, 4e-6, 6e-6]
cpp_axis = np.logspace(-6.5,-5,50)

# For plot_efficiency_vs_p
KD_vals_2 = [1e-6, 5e-6, 1e-5, 5e-5]
cpp_axis_2 = np.logspace(-5.2, -0.5)

# For plot_compare_fluxes
cpp_vals_2 = [1e-7, 1e-5]

#### MAIN CALLS ####

param = Params8(rD, rp, rt, cDo, cpc, vD_list, vp_list) # Create instantiation of Params8 object

plot_efflux_vs_KD(param, KD_axis, Kp_list, V_base, kappa, cDc, cpp_vals)
plot_efflux_vs_D(param, KD_list, Kp_list, V_base, kappa, cDc_axis, cpp_vals)
# plot_KM(param, KD_vals, Kp_list, V_base, kappa, cDc_axis, cpp_axis)
# plot_efficiency_vs_p(param, KD_vals_2, Kp_list, V_base, kappa, cDc, cpp_axis_2)
# plot_compare_fluxes(param, KD_axis, Kp_list, V_base, kappa, cDc, cpp_vals_2)