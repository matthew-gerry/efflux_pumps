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

def plot_efflux_vs_KD(param, KD_axis, Kp_list, KD_ratio, Qp_list, V_base, kappa, cDc, cpp_vals):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY '''

    efflux_vals = len(cpp_vals)*[[]] # List to populate with values of the mean efflux

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        for j in range(len(KD_axis)):
            KD = KD_axis[j]
        
            efflux_output = pump.efflux_numerical_7(param, KD, Kp_list, KD_ratio*KD, Qp_list, V_base, kappa, cDc, cpp)

            efflux_vals[i] = efflux_vals[i] + [efflux_output]
        
    # Configure some things for plotting
    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting
    KD_axis_uM = microfy(KD_axis) # KD_axis in uM
    cpp_vals_uM = microfy(cpp_vals)

    for i in range(len(cpp_vals)):
        efflux_plot = [x/param.rD for x in efflux_vals[i]]
        plt.semilogx(KD_axis_uM, efflux_plot,label="$[p]_{per} = "+str(round(cpp_vals_uM[i],1))+"\:\mu M$", linestyle = ls_list[i])
    plt.xlim([min(KD_axis_uM),max(KD_axis_uM)])
    plt.ylim([0, 4.5e-6])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel(r"$J\nu_D/k_D^+$")
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    plt.text(0.03, 3.9e-6, 'A', fontsize=16)
    plt.legend()
    plt.show()

    
def plot_efficiency_vs_p(param, KD_axis, Kp_list, KD_ratio, Qp_list, V_base, kappa, cDc, cpp_vals):
    ''' PLOT CHEMICAL EFFICIENCY AS A FUNCTION OF KD FOR VARIOUS [p] VALUES, KD RATIO FIXED '''

    efficiency = len(cpp_vals)*[[]] # Allocate array for chemical efficiency values

    # Evaluate efflux at each value of periplasmic proton concentration
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        for j in range(len(KD_axis)):
            KD = KD_axis[j]
            
            efflux = pump.efflux_numerical_7(param, KD, Kp_list, KD_ratio*KD, Qp_list, V_base, kappa, cDc, cpp)
            p_flux = pump.p_flux_7(param, KD, Kp_list, KD_ratio*KD, Qp_list, V_base, kappa, cDc, cpp)
            efficiency[i] = efficiency[i] + [efflux/p_flux] # Chemical efficiency

    # Configure some things for plotting
    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting
    KD_axis_uM = microfy(KD_axis) # Kp_vals in uM
    cpp_vals_uM = microfy(cpp_vals) # cpp_axis in uM

    plt.figure()
    for i in range(len(cpp_vals)):
        plt.semilogx(KD_axis_uM, efficiency[i],label="$[p]_{per} =\:"+str(round(cpp_vals_uM[i],1))+"\:\mu M$",linestyle=ls_list[i])
    plt.xlim([min(KD_axis_uM),max(KD_axis_uM)])
    plt.ylim([0, 0.64])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel("$J/J_p$")
    # plt.legend()
    plt.text(0.03,0.565,'B',fontsize=16)
    plt.show()


def plot_epr(param, KD, Kp_list, KD_ratio, Qp_list, V_base, kappa, cDc, cpp_axis):
    ''' PLOT THE ENTROPY PRODUCTION RATE AS A FUNCTION OF THE PERIPLASMIC PROTON CONCENTRATION '''

    sigma5, sigma7 = [], [] # Allocate array for entropy production rate lists

    # Evaluate EPR at each value of the periplasmic proton concentration
    for i in range(len(cpp_axis)):
        cpp = cpp_axis[i]
        sigma5_val = pump.entropy_5(param, KD, Kp_list[0], KD_ratio, Qp_list[0], V_base, kappa, cDc, cpp)
        sigma7_val = pump.entropy_7(param, KD, Kp_list, KD_ratio, Qp_list, V_base, kappa, cDc, cpp)
        
        sigma5 = sigma5 + [sigma5_val]
        sigma7 = sigma7 + [sigma7_val]

    V_entropy = -q*V_base/T

    sigma7_plot = [kB*S/V_entropy/param.rD for S in sigma7]
    sigma5_plot = [kB*S/V_entropy/param.rD for S in sigma5]
    cpp_axis_uM = [1e6*cpp for cpp in cpp_axis]

    # Configure some things for plotting
    ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting

    fig, ax = plt.subplots()

    ax.semilogx(cpp_axis_uM, sigma7_plot, label="Seven-state model", color="purple", linestyle=ls_list[0])
    ax.semilogx(cpp_axis_uM, sigma5_plot, label="Five-state model", color="olive", linestyle=ls_list[1])
    ax.set_xlim([min(cpp_axis_uM), max(cpp_axis_uM)])
   
    # Use scalar formatter to be able to set ticklabel format to plain
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='plain', scilimits=(0,0), useMathText=True)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
   
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    ax.set_xlabel("$[p]_{per}\;(\mu M)$")
    ax.set_ylabel("$\dot{\Sigma}$")
    ax.legend(loc="upper right")
    ax.text(0.13,1.45e-4,'C',fontsize=16)
    plt.show()




#### GLOBAL VARIABLES ####

# Parameter values
rD = 1e6 # 1/s
rp = 1e14 # 1/s
rt = 1e6 # 1/s, unlike in the three-state model, rt does not depend on other char. rates
vD = 1 # 1/M
vp = 1e-6 # 1/M
cDo = 1e-5 # M
cpc = 1e-7 # M

KD = 1e-6
Kp_pump = 1e-6 # M, primary proton binding affinity for pumping cycle
Kp_waste = 1e-6 # M, primary proton binding affinity for waste cycle
V_base = -np.log(100)*kB*T/q # V, base voltage, about -110 mV
# kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
kappa = 0
cDc = 1e-5 # M, cytoplasmic drug concentration (except plot_efflux_vs_D)

# Plot axis and parameters defining different curves
KD_axis = np.logspace(-8, -2, 200)
Kp_list = [Kp_pump, Kp_waste]
Qp_list = Kp_list
cpp_vals = [1e-7, 3e-7, 6e-7, 1e-6]
KD_ratio = 10

# For plot_epr
cpp_axis = np.logspace(-7,-5, 200)

#### MAIN CALLS ####

param = Params3(rD, rp, rt, cDo, cpc, vD, vp) # Create instantiation of Params3 object

# plot_efflux_vs_KD(param, KD_axis, Kp_list, KD_ratio, Qp_list, V_base, kappa, cDc, cpp_vals)
# plot_efficiency_vs_p(param, KD_axis, Kp_list, KD_ratio, Qp_list, V_base, kappa, cDc, cpp_vals)
plot_epr(param, KD, Kp_list, KD_ratio, Qp_list, V_base, kappa, cDc, cpp_axis)