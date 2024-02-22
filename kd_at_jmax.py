'''
kd_at_jmax.py

Efflux pumps project: code to numerically approximate the value of K_D
at which the efflux curve is peaked, for the 3-state and 8-state 
models.

Matthew Gerry, March 2023
'''

import numpy as np
from os.path import exists
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump


#### FUNCTIONS: GENERAL ####

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

def get_logwidth(efflux_data, KD_axis):
    '''
    NUMERICALLY APPROXIMATE THE LOG-WIDTH IN KD OF THE EFFLUX VS KD CURVE.

    GENERATE MATRIX TO USE AS THE ARGUMENT efflux_data USING THE FUNCTION efflux_matrix_3 or efflux_matrix_8.
    YOU MAY CHOOSE TO SAVE THIS DATA IN A .npy FILE FOR RE-USE.    
    '''

    J_logwidth = np.zeros([np.shape(efflux_data)[2], np.shape(efflux_data)[0]]) # Initialize array for log-width values

    for j in range(np.shape(efflux_data)[2]): 
        for i in range(np.shape(efflux_data)[0]):
            J_at_cpp = efflux_data[i,:,j] # Efflux as a function of KD at fixed cpp
            # J_halfmax_at_cpp = 0.5*J_at_cpp.max()
            J_halfmax_at_cpp = 1

            arg1, arg2 = 0,0 # Initialize args for positions of half max
            for k in range(len(J_at_cpp)-1):
                if J_at_cpp[k] < J_halfmax_at_cpp and J_at_cpp[k+1] >= J_halfmax_at_cpp:
                    arg1 = k
                if J_at_cpp[k] > J_halfmax_at_cpp and J_at_cpp[k+1] <= J_halfmax_at_cpp:
                    arg2 = k

            J_logwidth[j,i] = np.log10(KD_axis[arg2]/KD_axis[arg1]) # Log-width of efflux vs K_D curve

    return J_logwidth


#### FUNCTIONS: DATA GENERATION FOR EACH MODEL ####

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

def efflux_matrix_5(param, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc_vals, cpp_axis, reversed_unbinding=True):
    ''' CALCULATE THE MATRIX OF EFFLUX VALUES WITH VARYING KD, CPP, UNICYCLIC WITH SEQUENTIAL UNBINDING '''

    efflux_vals = np.zeros([len(cpp_axis),len(KD_axis),len(cDc_vals)]) # Initialize matrix to store efflux vals

    # Evaluate mean efflux at each KD-cpp pair, for each drug concentration
    for j in range(len(cDc_vals)):
        cDc = cDc_vals[j]

        for i in range(len(cpp_axis)):
            cpp = cpp_axis[i]

            # Efflux as a function of KD at set cpp and cDc
            efflux_vals[i,:,j] = np.vectorize(pump.efflux_numerical_5)(param, KD_axis, Kp, KD_ratio*KD_axis, Kp_ratio*Kp, V_base, kappa, cDc, cpp, reversed_unbinding)
      
    return efflux_vals

def efflux_matrix_7(param, KD_axis, Kp_list, KD_ratio, Kp_ratio, V_base, kappa, cDc_vals, cpp_axis):
    ''' CALCULATE THE MATRIX OF EFFLUX VALUES WITH VARYING KD, CPP, MULTICYCLIC WITH FUTILE CYCLE '''

    Qp_list = [Kp_list[0]*Kp_ratio, Kp_list[1]*Kp_ratio] # Derive Qp values from Kp ratio and Kp list

    efflux_vals = np.zeros([len(cpp_axis),len(KD_axis),len(cDc_vals)]) # Initialize matrix to store efflux vals

    # Evaluate mean efflux at each KD-cpp pair, for each drug concentration
    for j in range(len(cDc_vals)):
        cDc = cDc_vals[j]

        for i in range(len(cpp_axis)):
            cpp = cpp_axis[i]

            # Efflux as a function of KD at set cpp and cDc
            for k in range(len(KD_axis)):
                KD = KD_axis[k]
                efflux_vals[i,k,j] = pump.efflux_numerical_7(param, KD, Kp_list, KD_ratio*KD, Qp_list, V_base, kappa, cDc, cpp)
      
    return efflux_vals

def efflux_matrix_p_ind(param, KD_axis, cDc_axis, kp_const):
    ''' MATRIX OF EFFLUX VALUES WITH VARYING KD, [D]_in, P=INDEPENDENT MODEL '''

    efflux_vals = np.zeros([len(cDc_axis),len(KD_axis),1]) # Initialize matrix to store efflux vals

    # Evaluate mean efflux at eacj KD-[D]_in pair
    for i in range(len(cDc_axis)):
        cDc = cDc_axis[i]

        # Efflux as a function of KD at set cDc
        efflux_vals[i,:,0] = np.vectorize(pump.efflux_numerical_p_ind)(param, KD_axis, cDc, kp_const)

    return efflux_vals



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

#### FUNCTIONS: PLOTTING - EACH TAKES A DATASET AS AN ARGUMENT ####

def plot_KD_at_Jmax(filename, KD_axis, cpp_axis):
    ''' PLOT KD AT Jmax FOR THE ANY MODEL, GIVEN APPROPRIATE DATA '''

    # Note data is saved in/loaded from the parent directory
    J = np.load("../"+filename+".npy")
    '''
    IF LOADING DATA, ENSURE THAT KD_axis AND cpp_axis FED INTO THIS FUNCTION
    MATCH THOSE USED TO CALCULATE DATA
    '''

    KD_at_Jmax = get_KD_at_Jmax(J, KD_axis)

    # Plot KD at Jmax
    fig, ax = plt.subplots()
    for j in range(np.shape(KD_at_Jmax)[0]): # Plot the KD at Jmax curves for different cDc values
        ax.loglog(1e6*cpp_axis, 1e6*KD_at_Jmax[j,:], label="$[D]_{in} = $"+str(int(1e6*cDc_vals[j]))+" $\mu M$", linestyle=ls_list[j])
    
    ax.set_xlim([1e6*min(cpp_axis), 1e6*max(cpp_axis)])
    # ax.set_ylim([1,15])

    # Use scalar formatter to be able to set ticklabel format to plain
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticks([1, 2, 5, 10, 20])

    ax.ticklabel_format(style='plain') # No scientific notation
    ax.set_xlabel("$[p]_{per}$ $(\mu M)$")
    ax.set_ylabel("$K_D$ at $J_{max}$ $(\mu M)$")    
    ax.text(0.15,13,"B",fontsize=18)
    plt.legend(loc="lower right")
    plt.show()

def plot_logwidth(filename, KD_axis, cpp_axis):
    ''' PLOT THE LOG WIDTH OF THE EFFLUX VS KD CURVE FOR ANY MODEL, GIVEN APPROPRIATE DATA '''

    # Note data is saved in/loaded from the parent directory
    J = np.load("../"+filename+".npy")
    '''
    ENSURE THAT KD_axis AND cpp_axis FED INTO THIS FUNCTION
    MATCH THOSE USED TO CALCULATE DATA
    '''

    J_logwidth = get_logwidth(J, KD_axis)

    # Plot log-width
    fig, ax = plt.subplots()
    for j in range(np.shape(J_logwidth)[0]): # Plot the KD at Jmax curves for different cDc values
        ax.semilogx(1e6*cpp_axis, J_logwidth[j,:], 'o', label="$[D]_{in} = $"+str(int(1e6*cDc_vals[j]))+" $\mu M$")
    
    # Use scalar formatter to be able to set ticklabel format to plain
    # ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    # ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])

    ax.ticklabel_format(style='plain') # No scientific notation
    ax.set_xlabel("$[p]_{per}$ $(\mu M)$")
    ax.set_ylabel("Log-width of $J$ with respect to $K_D$")    
    plt.legend()
    plt.show()


def plot_logwidth_p_ind(filename, KD_axis, cDc_axis):
    ''' PLOT THE LOG WIDTH OF THE EFFLUX VS KD CURVE AS A FUNCTON OF THE INSIDE DRUG CONCENTRATION, INTENDED FOR p-INDEPENDENT MODEL '''

    # Note data is saved in/loaded from the parent directory
    J = np.load("../"+filename+".npy")

    '''
    ENSURE THAT KD_axis AND cDc_axis FED INTO THIS FUNCTION
    MATCH THOSE USED TO CALCULATE DATA
    '''

    J_logwidth = get_logwidth(J, KD_axis)

    # Plot log-width
    fig, ax = plt.subplots()
    for j in range(np.shape(J_logwidth)[0]): # Plot the KD at Jmax curves for different cDc values
        ax.semilogx(1e6*cDc_axis, J_logwidth[j,:])
    
    # Use scalar formatter to be able to set ticklabel format to plain
    # ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    # ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    # ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xlim([1e6*min(cDc_axis), 1e6*max(cDc_axis)])
    ax.set_ylim([0,6.8])

    ax.ticklabel_format(style='plain') # No scientific notation
    ax.set_xlabel("$[D]_{in}$ $(\mu M)$")
    ax.set_ylabel("Log-width of $J$ with respect to $K_D$")    
    plt.show()


def plot_compare_logwidth(filenameA, filenameB, KD_axis, cpp_axis, models):
    ''' PLOT THE LOG WIDTH OF EFFLUX VS KD CURVE FOR BOTH MODELS, GIVEN DATA FILES FOR EACH '''
    
    # Note data is saved in/loaded from the parent directory
    JA = np.load("../"+filenameA+".npy")
    JB = np.load("../"+filenameB+".npy")

    '''
    ENSURE THAT KD_axis AND cpp_axis FED INTO THIS FUNCTION
    MATCH THOSE USED TO CALCULATE DATA
    '''
    
    # Get log-width data from each data set
    J_logwidth_A = get_logwidth(JA, KD_axis)
    J_logwidth_B = get_logwidth(JB, KD_axis)

    # Plot log-width - we cycle through each cDc value, though in practice only one is used
    fig, ax = plt.subplots()
    for j in range(np.shape(J_logwidth_A)[0]): # Plot the KD at Jmax curves for different cDc values
        ax.semilogx(1e6*cpp_axis, J_logwidth_A[j,:], label=models[0]+"-state model", color="purple", linestyle=ls_list[0])
        ax.semilogx(1e6*cpp_axis, J_logwidth_B[j,:], label=models[1]+"-state model", color="olive", linestyle=ls_list[1])

       # Use scalar formatter to be able to set ticklabel format to plain
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_xlim([min(1e6*cpp_axis), max(1e6*cpp_axis)])

    ax.ticklabel_format(style='plain') # No scientific notation
    ax.set_xlabel("$[p]_{per}$ $(\mu M)$")
    ax.set_ylabel("Log-width of $J$ with respect to $K_D\;(\mu M)$")
    # ax.text(0.15, 3.78, "D", fontsize=16)
    plt.legend()
    plt.show()


def plot_contour(filename, KD_axis, cpp_axis, rD):
    ''' PLOTS EFFLUX AS A FUNCTION OF BOTH [p] AND KD ON A CONTOUR PLOT USING DATA GENERATED WITH "EFFLUX_MATRIX" FUNCTION '''

    # Load the data, break out of function execution if not present
    try:
        J = np.load("../"+filename+".npy")
        ''' ENSURE THAT rD, KD_axis AND cpp_axis FED INTO THIS FUNCTION MATCH THOSE USED TO CALCULATE THE DATA '''
    except:
        print(filename+".npy not found in parent directory. Generate data using efflux_matrix_5")
        return

    KD_at_Jmax = get_KD_at_Jmax(J, KD_axis) # Get KD_at_Jmax to superpose on map

    KD_micro = 1e6*KD_axis
    cpp_micro = 1e6*cpp_axis

    # Prepare grid for plotting
    [X,Y] = np.meshgrid(KD_micro, cpp_micro)

    fig, ax = plt.subplots()

    # Note the efflux data array may have multiple layers for multiple [D]_in values
    # We set the index corresponding to this to 1, hardcoded in (next two lines)
    sctr = ax.scatter(X,Y,c=J[:,:,1]/rD,marker='x')
    sctr2 = ax.plot(1e6*KD_at_Jmax[1,:],cpp_micro,'-.k')
    cbar = fig.colorbar(sctr)
    cbar.formatter.set_useMathText(True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(min(KD_micro),max(KD_micro))
    ax.set_xlim(5e-2, 1e4)
    ax.set_ylim(min(cpp_micro),max(cpp_micro))
    ax.set_xlabel("$K_D\;(\mu M)$")
    ax.set_ylabel("$[p]_{per}\;(\mu M)$")

    ax.text(1.8e-1,5.7,'B',fontsize=18,color='w')
    cbar.ax.set_ylabel(r"$J\nu_D/k_D^+$")

    plt.show()


#### GLOBAL VARIABLES ####

ls_list = [(0,(1,1)), "dashdot", "dashed", (0,(3,1,1,1,1,1))] # Linestyle list, for plotting

# Parameter values
rD = 1e8 # 1/s
rp = 1e14 # 1/s
rt3 = 1e18 # 1/s
rt5 = 1e6 # 1/s - rt differs for 3 and 5 state models for physical consistency
vD = 1 # 1/M
vp = 1e-6 # 1/M
cDo = 1e-5 # M
cDoB = 1e-7 # M, for p-independent pump
cpc = 1e-7 # M

Kp = 1e-7 # M, proton binding affinity (all models)
Kp_list = [Kp, Kp] # M, use equal value for both elements of Kp_list (for seven-state model)
Kp_ratio = 1 #  Ratio of Kp from outside to inside
KD_ratio = 10 # Ratio of KD from outside to inside
kp_const = rp*vp*Kp # Specific to p_independent model

V_base = -np.log(100)*kB*T/q # V, base voltage, about -110 mV 
# kappa = -0.028 # V, voltage dependence on pH difference across the inner membrane
kappa = 0

# Axes and values for comutations, plotting of KD_at_Jmax
KD_axis = np.logspace(-9, 2.5, 1500) # M, drug binding affinity
KD_axis_B = np.logspace(-11, 0.5, 1000) # M, drug binding affinity - lower KD needed to probe p-independent model
cpp_axis = np.logspace(-7,-5, 400) # M, periplasmic proton concentration
cDc_vals = np.array([1e-6, 1e-5]) # M, cytoplasmic drug concentration
cDc_axis = np.logspace(-6.5, -5, 500)

# File names for 3- and 5-state efflux data to be stored
filename_map_3 = "J_map_3"
filename_map_5 = "J_map_5"
filename_map_p_ind = "J_map_p_ind"

# Setup of the log-width comparison
cDc_vals_wc = np.array([1e-5]) # M, cytoplasmic drug concentration (just choose one value)

filename_compare_3 = "J_compare_3"
filename_compare_5 = "J_compare_5"
filename_compare_7 = "J_compare_7"

#### MAIN CALLS ####

param3 = Params3(rD, rp, rt3, cDo, cpc, vD, vp) # Create instantiation of Params3 class to use for 3-state model plots
param5 = Params3(rD, rp, rt5, cDo, cpc, vD, vp) # And one for 5-state model plots
param3B = Params3(rD, rp, rt3, cDoB, cpc, vD, vp) # 3-state param object with lower [D]_out for p-independent plotting

plots_3state = False
if plots_3state:
    # Prepare data for 5-state model contour plot if necessary, then create plot
    data_exists = exists("../"+filename_map_3+".npy")

    if not data_exists:
        J_map_3 = efflux_matrix_3(param3, KD_axis, Kp, V_base, kappa, cDc_vals, cpp_axis)
        np.save("../"+filename_map_3, J_map_3)
    
    plot_KD_at_Jmax(filename_map_3, KD_axis, cpp_axis)
    plot_logwidth(filename_map_3, KD_axis, cpp_axis)
    # plot_contour(filename_map_3, KD_axis, cpp_axis, rD)


plots_5state = False
if plots_5state:
    # Prepare data for 5-state model contour plot if necessary, then create plot
        
    data_exists = exists("../"+filename_map_5+".npy")
    
    if not data_exists:
        J_map_5 = efflux_matrix_5(param5, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc_vals, cpp_axis, reversed_unbinding=True)
        np.save("../"+filename_map_5, J_map_5)
    
    plot_KD_at_Jmax(filename_map_5, KD_axis, cpp_axis)
    # plot_logwidth(filename_map_5, KD_axis, cpp_axis)
    plot_contour(filename_map_5, KD_axis, cpp_axis, rD)


plots_p_ind = False
if plots_p_ind:

    data_exists = exists("../"+filename_map_p_ind+".npy")

    if not data_exists:
        J_map_p_ind = efflux_matrix_p_ind(param3B, KD_axis_B, cDc_axis, kp_const)
        np.save("../"+filename_map_p_ind, J_map_p_ind)
    
    plot_logwidth_p_ind(filename_map_p_ind, KD_axis_B, cDc_axis)


plot_width_comparison = False
if plot_width_comparison:
    # Prepare data for both models if necessary, then create plot

    data3_exists = exists("../"+filename_compare_3+".npy")
    data5_exists = exists("../"+filename_compare_5+".npy")
    data7_exists = exists("../"+filename_compare_7+".npy")

    if not data3_exists:
        J_compare_3 = efflux_matrix_3(param3, KD_axis, Kp, V_base, kappa, cDc_vals_wc, cpp_axis)
        np.save("../"+filename_compare_3+".npy", J_compare_3)

    if not data5_exists:
        J_compare_5 = efflux_matrix_5(param5, KD_axis, Kp, KD_ratio, Kp_ratio, V_base, kappa, cDc_vals_wc, cpp_axis, reversed_unbinding=True)
        np.save("../"+filename_compare_5+".npy", J_compare_5)
    
    if not data7_exists:
        J_compare_7 = efflux_matrix_7(param5, KD_axis, Kp_list, KD_ratio, Kp_ratio, V_base, kappa, cDc_vals_wc, cpp_axis)
        np.save("../"+filename_compare_7+".npy", J_compare_7)

    plot_compare_logwidth(filename_compare_3, filename_compare_5, KD_axis, cpp_axis, ["Three", "Five"])
    plot_compare_logwidth(filename_compare_7, filename_compare_5, KD_axis, cpp_axis, ["Seven", "Five"])


# Output logwidth for proton-independent model
