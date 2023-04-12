'''
kd_at_jmax.py

Efflux pumps project: code to numerically approximate the value of K_D
at which the efflux curve is peaked, for the 3-state and 8-state 
models.

Matthew Gerry, March 2023
'''

import numpy as np
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump

#### GLOBAL VARIABLES ####

ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting

Kp = 1e-6 # M, proton binding affinity (three-state model)

KDA = 1e-6 # M, drug binding affinity for cycle A (eight-state model)
Kp_list = [1e-6, 1e-6, 1e-6, 1e-6] # M, proton binding affinities (eight-state model)

V_base = -0.15 # V, base voltage
kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane

# Axes and values for comutations, plotting
KD_axis = np.logspace(-5,-1,1000) # M, drug binding affinity
cpp_axis = np.logspace(-7,-5,200) # M, periplasmic proton concentration
cDc_vals = np.array([1e-6, 1e-5]) # M, cytoplasmic drug concentration


#### FUNCTION: GENERAL ####

def get_KD_at_Jmax(efflux_data, KD_axis):
    ''' 
    NUMERICALLY GET THE KD VALUE AT JMAX FOR EACH PROTON CONCENTRATION.
    
    GENERATE MATRIX TO USE AS THE ARGUMENT efflux_data USING THE FUNCTION efflux_matrix_3 or efflux_matrix_8.
    YOU MAY CHOOSE TO SAVE THIS DATA IN A .npy FILE FOR RE-USE.    
    '''

    KD_at_Jmax = np.zeros([np.shape(efflux_data)[2], np.shape(efflux_data)[0]]) # Initialize array for KD at Jmax values
    for j in range(np.shape(efflux_data)[2]): 
        for i in range(np.shape(efflux_data)[0]):
            J_at_cpp = efflux_data[i,:,j] # Efflux as a function of KD at fixed cpp
            index_of_max = J_at_cpp.argmax() # Identify index of max efflux in list
            KD_at_Jmax[j, i] = KD_axis[index_of_max] # Value of KD for which max efflux is achieved

    return KD_at_Jmax


#### FUNCTIONS: THREE-STATE MODEL ####

def efflux_matrix_3(param, KD_axis, Kp, V_base, kappa, cDc_vals, cpp_axis):
    ''' CALCULATE THE MATRIX OF EFFLUX VALUES WITH VARYING KD, CPP, UNICYCLIC '''

    efflux_vals = np.zeros([len(cpp_axis),len(KD_axis),len(cDc_vals)]) # Initialize matrix to store efflux vals

    # Evaluate mean efflux at each KD-cpp pair, for each drug concentration
    for j in range(len(cDc_vals)):
        cDc = cDc_vals[j]

        for i in range(len(cpp_axis)):
            cpp = cpp_axis[i]

            # Efflux as a function of KD at set cpp and cDc
            efflux_vals[i,:,j] = np.vectorize(pump.efflux_numerical_3)(param, KD_axis, Kp, V_base, kappa, cDc, cpp)
      
    return efflux_vals


def plot_KD_at_Jmax_3(param, KD_axis, Kp, V_base, kappa, cDc_vals, cpp_axis, filename):
    ''' PLOT KD AT Jmax FOR THE THREE STATE MODEL AT THE PARAMETER VALUES SPECIFIED '''

    # Note data is saved in/loaded from the parent directory
    try: # Load data if saved
        J = np.load("../"+filename+".npy")
        '''
        IF LOADING DATA, ENSURE THAT KD_axis AND cpp_axis FED INTO THIS FUNCTION
        MATCH THOSE USED TO CALCULATE DATA
        '''

    except: # Otherwise calculate the efflux values
        J = efflux_matrix_3(param, KD_axis, Kp, V_base, kappa, cDc_vals, cpp_axis)
        np.save("../"+filename+".npy", J) # Save data for next time (good if just playing around with plot formatting, bad if changing param values on consecuative runs)

    KD_at_Jmax = get_KD_at_Jmax(J, KD_axis)

    # Plot KD at Jmax
    fig, ax = plt.subplots()
    for j in range(np.shape(KD_at_Jmax)[0]): # Plot the KD at Jmax curves for different cDc values
        ax.loglog(1e6*cpp_axis, 1e3*KD_at_Jmax[j,:], label="$[D]_{cyt} = $"+str(int(1e6*cDc_vals[j]))+" $\mu M$", linestyle=ls_list[j])
    
    # Use scalar formatter to be able to set ticklabel format to plain
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])

    ax.ticklabel_format(style='plain') # No scientific notation
    ax.set_xlabel("$[p]_{per}$ $(\mu M)$")
    ax.set_ylabel("$K_D$ at $J_{max}$ $(mM)$")    
    plt.legend()
    plt.show()


#### FUNCTIONS: EIGHT-STATE MODEL ####

def efflux_matrix_8(param, KD_axis, Kp_list, V_base, kappa, cDc_vals, cpp_axis):
    ''' CALCULATE THE MATRIX OF EFFLUX VALUES WITH VARYING KD, CPP, EIGHT-STATE MODEL '''

    efflux_vals = np.zeros([len(cpp_axis),len(KD_axis),len(cDc_vals)]) # Initialize matrix to store efflux vals

    # Evaluate mean efflux at each KD-cpp pair, for each drug concentration
    for j in range(len(cDc_vals)):
        cDc = cDc_vals[j]

        for i in range(len(cpp_axis)):
            cpp = cpp_axis[i]

            for k in range(len(KD_axis)): # Must write for-loop explicitly due to how KD_list is defined
                KD_list = 2*[KD_axis[k]]

                efflux_vals[i,k,j] = pump.efflux_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)
    
    return efflux_vals


def plot_KD_at_Jmax_8(param, KD_axis, Kp_list, V_base, kappa, cDc_vals, cpp_axis, filename):
    ''' PLOT KD AT Jmax FOR THE EIGHT-STATE MODEL AT THE PARAMETER VALUES SPECIFIED '''

    # Note data is saved in/loaded from the parent directory
    try: # Load data if saved
        J = np.load("../"+filename+".npy")
        '''
        IF LOADING DATA, ENSURE THAT KD_axis AND cpp_axis FED INTO THIS FUNCTION
        MATCH THOSE USED TO CALCULATE DATA
        '''

    except: # Otherwise calculate the efflux values
        J = efflux_matrix_8(param, KD_axis, Kp_list, V_base, kappa, cDc_vals, cpp_axis)
        np.save("../"+filename+".npy", J) # Save data for next time (good if just playing around with plot formatting, bad if changing param values on consecuative runs)

    KD_at_Jmax = get_KD_at_Jmax(J, KD_axis)

    # Plot KD at Jmax
    fig, ax = plt.subplots()
    for j in range(np.shape(KD_at_Jmax)[0]): # Plot the KD at Jmax curves for different cDc values
        ax.loglog(1e6*cpp_axis, 1e3*KD_at_Jmax[j,:], label="$[D]_{cyt} = $"+str(int(1e6*cDc_vals[j]))+" $\mu M$", linestyle=ls_list[j])
    
    # Use scalar formatter to be able to set ticklabel format to plain
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])

    ax.ticklabel_format(style='plain') # No scientific notation
    ax.set_xlabel("$[p]_{per}$ $(\mu M)$")
    ax.set_ylabel("$K_D$ at $J_{max}$ $(mM)$")    
    plt.legend()
    plt.show()


#### MAIN CALLS ####

param3 = Params3(1e6, 1e6, 1e6, 1e-11, 1e-7, 1, 0.1) # Create instantiation of Params3 class
plot_KD_at_Jmax_3(param3, KD_axis, Kp, V_base, kappa, cDc_vals, cpp_axis, "dummy_data3")

param8 = Params8(1e6, 1e6, 1e6, 1e-11, 1e-7, [1,1], [0.1,0.1,0.1,0.1]) # Create instantiation of Params8 class
plot_KD_at_Jmax_8(param8, KD_axis, Kp_list, V_base, kappa, cDc_vals, cpp_axis, "dummy_data8")



