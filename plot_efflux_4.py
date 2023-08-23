'''
plot_efflux_4.py

Plotting functions for the four-state model of bacterial efflux pumps.

This includes functions to numerically estimate KM and, in turn, the specificity,
since we do not have analytic expressions for these quantities as we do for the
three-state model. 

When calling functions, an object of the class Params4 should be passed to as 
argument denoted param.

Matthew Gerry, August 2023
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from parameters import *
import efflux_pumps as pump


#### FUNCTIONS: CALCULATIONS ####

def get_efflux_vs_D(param, KD, Kp_list, V_base, kappa, cDc_axis, cpp_vals):
    ''' GET A LIST OF LISTS HOLDING EFFLUX VS DRUG CONCENTRATION VALUES FOR VARIOUS PROTON CONCENTRATIONS '''

    efflux_vals = []

    # Evaluate mean efflux at each value of cpp and [D]
    for i in range(len(cpp_vals)):
        cpp = cpp_vals[i]

        efflux_at_fixed_cpp = []
        for j in range(len(cDc_axis)):
            cDc = cDc_axis[j]

            efflux_output = pump.efflux_numerical_4(param, KD, Kp_list, V_base, kappa, cDc, cpp)
            efflux_at_fixed_cpp.append(efflux_output)

        efflux_vals.append(efflux_at_fixed_cpp)
    
    return efflux_vals

def get_KM(param, KD, Kp_list, V_base, kappa, cDc_axis, cpp_axis):
    ''' USE NUMERICALLY CALCULATED EFFLUX VS [D] TO APPROXIMATE KM ([D] AT HALF MAX EFFLUX) '''

    KM_vals = np.zeros(len(cpp_axis)) # Allocate array for KM as a function of [p]

    efflux_matrix = get_efflux_vs_D(param, KD, Kp_list, V_base, kappa, cDc_axis, cpp_axis)

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