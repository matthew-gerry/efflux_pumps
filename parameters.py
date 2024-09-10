'''
parameters.py

Define the parameters for the three-, five-, and seven-state models of bacterial efflux pumps.
Include only the parameters that remain constant between different plots (interaction volumes, characteristic rates, etc.).

Matthew Gerry, March 2023
'''

import numpy as np

#### PHYSICAL CONSTANTS ####

kB = 1.38065e-23 # J/K, Boltzmann constant
T = 295 # K, temperature
q = 1.602e-19 # C, proton charge


#### CLASS DEFINITION ####

class Parameters:
    ''' PARAMETERS FOR THE EFFLUX PUMP MODELS '''
    
    # Physical constants - equate to global variables
    kB = kB; T = T; q = q

    def __init__(self, rD, rp, rt, cDo, cpc, vD, vp):
        
        # System-specific parameters
        self.rD = rD # 1/s, characteristic rate associated with drug binding
        self.rp = rp # 1/s, characteristic rate associated with proton binding
        self.rt = rt # 1/s, characteristic rate associated with protein conformational changes
        # For consistency, 1/rt = 1/rD + 1/rp + tau_C, tau_C a timescale for conformational changes
        self.cDo = cDo # M, outside drug concentration
        self.cpc = cpc # M, cytoplasmic proton concentration
        self.vD = vD # 1/M, drug interaction volume
        self.vp = vp # 1/M, proton interaction volume


#### DERIVED PARAMETERS ####

def get_derived_params(param, cpp, V_base, kappa):
    ''' 
    GET DERIVED PARAMETERS BASED ON INPUT VALUES OF PERIPLASMIC PROTON CONCENTRATION, ELECTRIC POTENTIALS.
    ARGUMENT param MAY BE EITHER A Params3 OR Params8 OBJECT.
    '''

    # Derived parameters
    delta_pH = np.log10(param.cpc/cpp) # pH difference between periplasm and cytoplasm
    V = V_base + kappa*delta_pH # Based on results of Lo et al., doi: 10.1529/biophysj.106.095265
    KG = np.exp(-param.q*V/(param.kB*param.T)) # KG as a function of the proton concentrations

    return delta_pH, V, KG
