'''
plot_efflux_7.py

Plotting functions for the seven-state model of bacterial efflux pumps.

This includes functions to numerically estimate KM and, in turn, the specificity,
since we do not have analytic expressions for these quantities as we do for the
three-state model. 

When calling functions, an object of the class Params3 should be passed to as 
argument denoted param.

Matthew Gerry, August 2023
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump


#### FUNCTIONS: CALCULATIONS ####

def get_efflux_vs_D(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc_axis, cpp_vals):
    ''' GET A LIST OF LISTS HOLDING EFFLUX VS DRUG CONCENTRATION VALUES FOR VARIOUS PROTON CONCENTRATIONS '''

    efflux_vals = []

    # Evaluate mean efflux at each value of cpp and [D]
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        efflux_at_fixed_cpp = []
        for j in range(len(cDc_axis)):
            cDc = cDc_axis[j]

            efflux_output = pump.efflux_numerical_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp)
            efflux_at_fixed_cpp.append(efflux_output)

        efflux_vals.append(efflux_at_fixed_cpp)
    
    return efflux_vals

def get_KM(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc_axis, cpp_axis):
    ''' USE NUMERICALLY CALCULATED EFFLUX VS [D] TO APPROXIMATE KM ([D] AT HALF MAX EFFLUX) '''

    KM_vals = np.zeros(len(cpp_axis)) # Allocate array for KM as a function of [p]

    efflux_matrix = get_efflux_vs_D(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc_axis, cpp_axis)

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

nanofy = lambda a : [1e9*x for x in a] # By 9 orders of magnitude
microfy = lambda a : [1e6*x for x in a] # By 6 orders of magnitude

def plot_efflux_vs_KD(param, KD_axis, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp_vals):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY '''

    efflux_vals = len(cpp_vals)*[[]] # List to populate with values of the mean efflux

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        for j in range(len(KD_axis)):
            KD = KD_axis[j]
        
            efflux_output = pump.efflux_numerical_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp)

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

    
def plot_efficiency_vs_p(param, KD_vals, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp_axis):
    ''' PLOT CHEMICAL EFFICIENCY AS A FUNCTION OF [p] FOR VARIOUS KD VALUES, KP RATIO FIXED '''

    efficiency = len(KD_vals)*[[]] # Allocate array for chemical efficiency values

    # Evaluate efflux at each value of periplasmic proton concentration
    for i in range(len(KD_vals)):
        KD = KD_vals[i]

        for j in range(len(cpp_axis)):
            cpp = cpp_axis[j]
            
            efflux = pump.efflux_numerical_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp)
            p_flux = pump.p_flux_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp)
            efficiency[i] = efficiency[i] + [efflux/p_flux] # Chemical efficiency

    # Configure some things for plotting
    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting
    KD_vals_uM = microfy(KD_vals) # Kp_vals in uM
    cpp_axis_uM = microfy(cpp_axis) # cpp_axis in uM

    plt.figure()
    for i in range(len(KD_vals)):
        plt.semilogx(cpp_axis_uM, efficiency[i],label="$K_D =\:"+str(int(KD_vals_uM[i]))+"\:\mu M$",linestyle=ls_list[i])
        plt.xlabel("$[p]_{per}\:(\mu M)$")
        plt.ylabel("$J/J_p$")
    plt.legend()
    plt.show()


#### GLOBAL VARIABLES ####

# Parameter values
rD = 1e8 # 1/s
rp = 1e7 # 1/s
rt = 1e7 # 1/s
vD = 1 # 1/M
vp = 1 # 1/M
cDo = 1e-5 # M
cpc = 1e-7 # M

KD = 1e-6
QD = 1e-6
Kp_pump = 1e-6 # M, primary proton binding affinity for pumping cycle
Kp_waste = 1e-6 # M, primary proton binding affinity for waste cycle
V_base = -0.15 # V, base voltage (except plot_specificity)
kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
cDc = 1e-5 # M, cytoplasmic drug concentration (except plot_efflux_vs_D)

# For plot_efflux_vs_KD
cpp_vals = [1e-7, 3e-7, 6e-7, 1e-6]
KD_axis = np.logspace(-7, 0, 100) # Also for plot_compare_fluxes

# For plot_efficiency_vs_p
KD_vals = [1e-6, 5e-6, 1e-5, 5e-5]
cpp_axis = np.logspace(-7, -1)
Kp_list = [Kp_pump, Kp_waste]
Qp_list = Kp_list

#### MAIN CALLS ####

param = Params3(rD, rp, rt, cDo, cpc, vD, vp) # Create instantiation of Params3 object

# plot_efflux_vs_KD(param, KD_axis, Kp_list, V_base, kappa, cDc, cpp_vals)
plot_efficiency_vs_p(param, KD_vals, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp_axis)