'''
pump_params.py

Define the parameters for the three- and eight-state models of bacterial efflux pumps.
Include only the parameters that remain constant between different plots (interaction volumes, r0, etc.).

Matthew Gerry, March 2023
'''

import numpy as np

#### PHYSICAL CONSTANTS ####
kB = 1.38065e-23 # J/K, Boltzmann constant
T = 295 # K, temperature
q = 1.602e-19 # C, proton charge

class Parameters:
    ''' PARAMETERS FOR THE EFFLUX PUMP MODELS '''
    
    # Physical constants - equate to global variables
    kB = kB; T = T; q = q

    def __init__(self, r0, cDo, cpc):
        
        # System-specific parameters
        self.r0 = r0 # 1/s, rate associated with protein conformational changes
        self.cDo = cDo # M, outside drug concentration
        self.cpc = cpc # M, cytoplasmic proton concentration


class Params3(Parameters):
    ''' PARAMETERS INCLUDING THOSE SPECIFIC TO THE THREE-STATE MODEL '''

    def __init__(self, r0, cDo, cpc, vD, vp):
        Parameters.__init__(self, r0, cDo, cpc)
        self.vD = vD # 1/M, drug interaction volume
        self.vp = vp # 1/M, proton interaction volume


class Params8(Parameters):
    ''' PARAMETERS INCLUDING THOSE SPECIFIC TO THE EIGHT-STATE MODEL '''

    def __init__(self, r0, cDo, cpc, vD_list, vp_list):
        Parameters.__init__(self, r0, cDo, cpc)
        self.vD_list = vD_list # 1/M, drug interaction volume (list of two values)
        self.vp_list = vp_list # 1/M, proton interaction volume (list of four values)

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