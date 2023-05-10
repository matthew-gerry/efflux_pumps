'''
efflux_nprotons.py

Explore the effects of changing the value of the parameter n, the number
of protons involved in the proton binding step of the efflux pump mechanism.

Presently we are raising Kp to the power of n as we do the proton concentration.
This is a choice; another option would be to keep Kp fixed value, though this
is not necessarily intuitive as its value will change regardless.

Matthew Gerry, May 2023
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump
import rate_matrix as rm


#### FUNCTIONS: THREE-STATE MODEL ####

def efflux_numerical_3_nprotons(param, KD, Kp, V_base, kappa, cDc, cpp, n):
    ''' GET MEAN EFFLUX RATE BY SOLVING FOR THE STEADY-STATE NUMERICALLY, 3-STATE MODEL '''
    cppn = cpp**n
    Kpn = Kp**n

    R = rm.rate_matrix_3(param, KD, Kpn, V_base, kappa, cDc, cppn)

    # Find steady state solution as eigenvector of R associated with the zero eigenvalue
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] # Efflux at steady state

    return efflux


def efflux_MM_3_nprotons(param, KD, Kp, V_base, kappa, cDc, cpp, n):
    ''' EFFLUX CALCULATED FROM THE MICHAELIS-MENTEN LIKE EXPRESSION DERIVED FOR THE REVERSIBLE CASE, THREE-STATE MODEL '''

    cppn = cpp**n
    Kpn = Kp**n
    KM = pump.KM_3(param, KD, Kpn, V_base, kappa, cppn) # Michaelis-Menten constant
    S = pump.spec_3(param, KD, Kpn, V_base, kappa, cDc, cppn) # Specificity

    J = cDc*KM*S/(cDc + KM) # efflux

    return J


#### PLOTTING FUNCTIONS ####

def plot_efflux_vs_KD_nprotons(param, KD_axis, Kp, V_base, kappa, cDc, cpp, n_vals):
    ''' MEAN EFFLUX AS A FUNCTION OF DRUG BINDING AFFINITY AT VARYING n'''

    mean_efflux = []

    # Evaluate efflux mean at each value of KD and cpp
    for i in range(len(n_vals)):
        n = n_vals[i]

        mean_output = np.vectorize(efflux_numerical_3_nprotons)(param, KD_axis, Kp, V_base, kappa, cDc, cpp, n)

        mean_efflux.append(mean_output)

    # Plot mean values and variances side by side
    KD_axis_uM = 1e6*KD_axis # KD_axis in uM
    # mean_efflux_nano = [1e9*x for x in mean_efflux] # mean efflux in nano s^-1
    for i in range(len(n_vals)):
        plt.loglog(KD_axis_uM, mean_efflux[i], label="$n = $"+str(round(n_vals[i],1)), linestyle = ls_list[i])
    plt.xlabel("$K_D\:(\mu M)$")
    plt.ylabel("$J\:(s^{-1})$")
    # plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
    plt.legend()
    plt.show()



#### GLOBAL VARIABLES ####

ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting


# Parameter values
rD = 1e6 # 1/s
rp = 1e6 # 1/s
rt = 1e6 # 1/s
vD = 1 # 1/M
vp = 0.1 # 1/M
cDo = 1e-11 # M
cpc = 1e-7 # M

# Variables
Kp = 1e-6 # M, proton binding affinity
V_base = -0.15 # V, base voltage (except plot_specificity)
kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
KD = 1e-5 # M, drug binding affininty (except plot_efflux_vs_KD)
cDc = 1e-5 # M, cytoplasmic drug concentration (except plot_efflux_vs_D)

# For plot_efflux_vs_KD_nprotons
KD_axis = np.logspace(-8, -0.5, 200) # M, drug binding affinity
n_vals = [1,2,3]
cpp = 5e-7 # M, cytoplasmic drug concentration


#### MAIN CALLS ####

param = Params3(rD, rp, rt, cDo, cpc, vD, vp) # Create instantiation of Params3 object
plot_efflux_vs_KD_nprotons(param, KD_axis, Kp, V_base, kappa, cDc, cpp, n_vals)