'''
efflux_pumps.py

All of the functions computing the efflux and related quantities for
bacterial efflux pumps using the three- and eight-state model.

Matthew Gerry, March 2023
'''

import numpy as np

from parameters import get_derived_params
import rate_matrix as rm

#### FUNCTIONS: THREE-STATE MODEL ####

def efflux_numerical_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    ''' GET MEAN EFFLUX RATE BY SOLVING FOR THE STEADY-STATE NUMERICALLY, 3-STATE MODEL '''
    R = rm.rate_matrix_3(param, KD, Kp, V_base, kappa, cDc, cpp)

    # Find steady state solution as eigenvector of R associated with the zero eigenvalue
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] # Efflux at steady state

    return efflux


def var_numerical_3(param, KD, Kp, V_base, kappa, cDc, cpp, dchi):
    ''' GET VARIANCE IN EFFLUX BY NUMERICALLY DIFFERENTIATING THE CGF, 3-STATE MODEL '''
    R = rm.rate_matrix_3(param, KD, Kp, V_base, kappa, cDc, cpp)
    
    # Do full counting statistics for the variance
    CGF = rm.cgf_3(R, dchi, 1) # Set chisteps to 1 since we just need the variance
    efflux_var = -np.diff(CGF, n=2)/(dchi**2) # Variance at steady state is given by the second derivative of the CGF wrt j*chi

    return efflux_var


def efflux_analytic_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    ''' MEAN AND VARIANCE EFFLUX RATE; EXPRESSIONS DERIVED BY HAND FROM FCS '''
    R = rm.rate_matrix_3(param, KD, Kp, V_base, kappa, cDc, cpp)

     # Coefficients in the characteristic polynomial
    a1 = np.trace(R)
    a2 = R[0,0]*R[1,1] + R[1,1]*R[2,2] + R[2,2]*R[0,0] - R[0,1]*R[1,0] - R[0,2]*R[2,0] - R[1,2]*R[2,1]

    # Derivatives of the coefficient a3 evaluated at chi=0
    a3_deriv = R[2,1]*R[1,0]*R[0,2] - R[0,1]*R[1,2]*R[2,0]
    a3_deriv2 = R[2,1]*R[1,0]*R[0,2] + R[0,1]*R[1,2]*R[2,0]

    # Expressions for the mean efflux and variance
    efflux = a3_deriv/a2
    efflux_var = (a3_deriv2 + 2*a1*efflux**2)/a2
    
    return efflux, efflux_var


def KM_3(param, KD, Kp, V_base, kappa, cpp):
    ''' THE MICHAELIS-MENTEN CONSTANT CHARACTERIZING SATURATION OF THE EFFLUX FOR THE THREE-STATE MODEL '''
    
    KG = get_derived_params(param, cpp, V_base, kappa)[2]   

    # Define ratios of characteristic rates
    RD = param.rD/param.rt
    Rp = param.rp/param.rt

    # Components of the MM-like expression
    Z = 1 + param.vD*param.vp*KD*Kp*KG # A partition function for states 1 and 3
    C1 = 1 + param.vD*KD*KG/Rp + param.vD*param.vp*KD*Kp*KG
    KM = (KD*(cpp*param.vp*KG/RD + C1) + param.cDo*param.cpc*(param.vD*KD/Rp + param.vp*(Kp + cpp)/RD)/Kp)/(cpp*Z/Kp + C1)

    # Approximate effective KM with experimentally relevant parameter ranges taken into account
    KM_simp = KD*(param.rt*param.vp*Kp*KG*cpp/(Kp*param.rD) + C1)/(cpp/Kp + C1)

    return KM, KM_simp


def spec_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    ''' SPECIFICITY OF THE PUMP (3-STATE MODEL) BASED ON MICHAELIS-MENTENT EXPRESSIONS '''
    
    KG = get_derived_params(param, cpp, V_base, kappa)[2]
    kt = param.rt*param.vD*param.vp*KD*Kp*KG/(1 + param.vD*param.vp*KD*Kp*KG) # Expression for the rotation transition rate

    # Define ratios of characteristic rates
    RD = param.rD/param.rt
    Rp = param.rp/param.rt

    Z = 1 + param.vD*param.vp*KD*Kp*KG # A partition function for states 1 and 3
    Kbeta = Kp*(1 + param.vD*KD*KG/(Rp*Z)) # Michaelis-Menten constants
    KM = KM_3(param, KD, Kp, V_base, kappa, cpp)[0]

    alpha = 1 - param.cDo*param.cpc/(cDc*cpp*KG) # Efficiency factor (1 in the irreversible limit)

    S = alpha*kt*cpp/(KM*(Kbeta + cpp)) # Specificity as defined in original manuscript, with alpha folded in
    
    return S
    

def efflux_MM_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    ''' EFFLUX CALCULATED FROM THE MICHAELIS-MENTEN LIKE EXPRESSION DERIVED FOR THE REVERSIBLE CASE, THREE-STATE MODEL '''

    KM = KM_3(param, KD, Kp, V_base, kappa, cpp)[0] # Michaelis-Menten constant
    S = spec_3(param, KD, Kp, V_base, kappa, cDc, cpp) # Specificity

    J = cDc*KM*S/(cDc + KM) # efflux

    return J


def entropy_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    ''' ENTROPY PRODUCTION RATE OF THE THREE-STATE MODEL '''

    R = rm.rate_matrix_3(param, KD, Kp, V_base, kappa, cDc, cpp)
    SS = rm.steady_state(R)
    SS = SS.reshape(3) # Flatten the steady state probability distribution

    # Entropy production is a sum over force-flux contributions derived from the rate matrix
    force = np.log(np.divide(R,np.transpose(R)))
    flux = np.multiply(R, np.array(3*[SS]))

    entropy_contributions = np.multiply(force, flux) # In units of kB
    EPR = entropy_contributions.sum() # Sum over all contributions

    return EPR


#### FUNCTIONS: EIGHT-STATE MODEL ####

def efflux_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp):
    ''' GET MEAN EFFLUX RATE BY NUMERICALLY SOLVING FOR THE STEADY STATE, 8-STATE MODEL '''
    
    R = rm.rate_matrix_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)

    # Find steady state solution as eigenvector of R
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] + SS[3]*R[4,3] - SS[4]*R[3,4] # Efflux at steady state

    return efflux


def var_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp, dchi):
    ''' GET VARIANCE IN THE EFFLUX BY NUMERICALLY DIFFERENTIATING THE CGF, 8-STATE MODEL '''

    R = rm.rate_matrix_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)

    # Do full counting statistics for the variance
    CGF = rm.cgf_8(R, dchi, 1) # Set chisteps to 1 since we just need the variance
    efflux_var = -np.diff(CGF, n=2)/(dchi**2) # Variance at steady state is given by the second derivative of the CGF wrt j*chi

    return efflux_var


def p_flux_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp):
    ''' GET PROTON FLUX RATE THROUGH A METHOD SIMILAR TO efflux_numerical_8 '''

    R = rm.rate_matrix_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)

    # Additional quantities needed
    KG = get_derived_params(param, cpp, V_base, kappa)[2]
    ktC = param.rt*param.vp_list[3]*Kp_list[3]*KG/(1 + param.vp_list[3]*Kp_list[3]*KG) # Rotation, cycle C

    # Find steady state solution as eigenvector of R
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] + SS[3]*R[4,3] - SS[4]*R[3,4] # Efflux at steady state
    p_flux = efflux + SS[7]*R[0,7] - SS[0]*R[7,0] + SS[5]*ktC - SS[4]*ktC*param.cpc/(Kp_list[3]*KG)

    return p_flux


def entropy_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp):
    ''' ENTROPY PRODUCTION RATE OF THE THREE-STATE MODEL '''

    R = rm.rate_matrix_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)
    SS = rm.steady_state(R)
    SS = SS.reshape(8) # Flatten the steady state probability distribution

    # Entropy production is a sum over force-flux contributions derived from the rate matrix
    force = np.log(np.divide(R,np.transpose(R)))
    flux = np.multiply(R, np.array(8*[SS]))

    entropy_contributions = np.multiply(force, flux) # In units of kB
    EPR = entropy_contributions.sum() # Sum over all contributions

    return EPR


#### FUNCTIONS: 5-STATE MODEL ####

def efflux_numerical_5(param, KD, Kp, QD, Qp, V_base, kappa, cDc, cpp, reversed_unbinding=False):
    ''' GET MEAN EFFLUX RATE BY NUMERICALLY SOLVING FOR THE STEADY STATE, 5-STATE MODEL '''
    
    if not reversed_unbinding:
        R = rm.rate_matrix_5(param, KD, Kp, QD, Qp, V_base, kappa, cDc, cpp)
    else:
        R = rm.rate_matrix_5a(param, KD, Kp, QD, Qp, V_base, kappa, cDc, cpp)

    # Find steady state solution as eigenvector of R
    SS = rm.steady_state(R)
    efflux = SS[4]*R[0,4] - SS[0]*R[4,0] # Efflux at steady state

    return efflux


def entropy_5(param, KD, Kp, KD_ratio, Qp, V_base, kappa, cDc, cpp):
    ''' ENTROPY PRODUCTION RATE OF THE THREE-STATE MODEL, WITH REVERSED UNBINDING '''

    R = rm.rate_matrix_5a(param, KD, Kp, KD*KD_ratio, Qp, V_base, kappa, cDc, cpp)
    SS = rm.steady_state(R)
    SS = SS.reshape(5) # Flatten the steady state probability distribution

    # Entropy production is a sum over force-flux contributions derived from the rate matrix
    # force = np.log(np.divide(R,np.transpose(R)))
    quotients = np.divide(R,np.transpose(R))
    quotients[np.isnan(quotients)] = 1 # Set 0/0 to 1 for consistency
    force = np.log(quotients)

    flux = np.multiply(R, np.array(5*[SS]))

    entropy_contributions = np.multiply(force, flux) # In units of kB
    EPR = entropy_contributions.sum() # Sum over all contributions

    return EPR

#### FUNCTIONS: 4-STATE MODEL ####

def efflux_numerical_4(param, KD, Kp_list, V_base, kappa, cDc, cpp):
    ''' GET MEAN EFFLUX RATE BY NUMERICALLY SOLVING FOR THE STEADY STATE, 4-STATE MODEL '''

    R = rm.rate_matrix_4(param, KD, Kp_list, V_base, kappa, cDc, cpp)

    # Find steady state solution as eigenvector of R
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] # Efflux at steady state

    return efflux

def p_flux_4(param, KD, Kp_list, V_base, kappa, cDc, cpp):
    ''' GET PROTON FLUX RATE THROUGH A METHOD SIMILAR TO efflux_numerical_4 '''

    R = rm.rate_matrix_4(param, KD, Kp_list, V_base, kappa, cDc, cpp)

    # Additional quantities needed
    KG = get_derived_params(param, cpp, V_base, kappa)[2]
    ktB = param.rt*param.vp_list[1]*Kp_list[1]*KG/(1 + param.vp_list[1]*Kp_list[1]*KG) # Rotation, waste cycle

    # Find steady state solution as eigenvector of R
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] # Efflux at steady state
    p_flux = efflux + SS[3]*ktB - SS[0]*ktB*param.cpc/(Kp_list[1]*KG)

    return p_flux


#### FUNCTIONS: 7-STATE MODEL ####

def efflux_numerical_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp):
    ''' NUMERICALLY CALCULATE THE EFFLUX FROM THE STEADY-STATE POPULATION DISTRUBTION FOR THE SEVEN-STATE MODEL '''

    R = rm.rate_matrix_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp)
    
    # Find steady state solution as eigenvector of R
    SS = rm.steady_state(R)
    efflux = SS[4]*R[0,4] - SS[0]*R[4,0] # Efflux at steady state

    return efflux

def p_flux_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp):
    ''' GET PROTON FLUX RATE THROUGH A METHOD SIMILAR TO efflux_numerical_7 '''

    R = rm.rate_matrix_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp)

    # Additional quantities needed
    KG = get_derived_params(param, cpp, V_base, kappa)[2]
    
    # Find steady state solution as eigenvector of R
    SS = rm.steady_state(R)
    efflux = SS[4]*R[0,4] - SS[0]*R[4,0] # Efflux at steady state
    p_flux = efflux + SS[6]*R[0,6] - SS[0]*R[6,0] # Add the proton flux from the waste cycle

    return p_flux

def entropy_7(param, KD, Kp_list, KD_ratio, Qp_list, V_base, kappa, cDc, cpp):
    ''' ENTROPY PRODUCTION RATE OF THE THREE-STATE MODEL '''

    R = rm.rate_matrix_7(param, KD, Kp_list, KD*KD_ratio, Qp_list, V_base, kappa, cDc, cpp)
    SS = rm.steady_state(R)
    SS = SS.reshape(7) # Flatten the steady state probability distribution

    # Entropy production is a sum over force-flux contributions derived from the rate matrix
    # force = np.log(np.divide(R,np.transpose(R)))
    quotients = np.divide(R,np.transpose(R))
    quotients[np.isnan(quotients)] = 1 # Set 0/0 to 1 for consistency
    force = np.log(quotients)

    flux = np.multiply(R, np.array(7*[SS]))

    entropy_contributions = np.multiply(force, flux) # In units of kB
    EPR = entropy_contributions.sum() # Sum over all contributions

    return EPR


#### FUNCTIONS: PROTON-INDEPENDENT MODEL ####

def efflux_numerical_p_ind(param, KD, cDc, kp_const):
    ''' GET MEAN EFFLUX RATE BY SOLVING FOR THE STEADY-STATE NUMERICALLY, 3-STATE MODEL '''
    R = rm.rate_matrix_p_ind(param, KD, cDc, kp_const)

    # Find steady state solution as eigenvector of R associated with the zero eigenvalue
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] # Efflux at steady state

    return efflux

def efflux_analytic_p_ind(param, KD, cDc, kp_const):
    ''' HAND-DERIVED EXPRESSION FOR THE EFFLUX RATE, ANALOGOUS TO THE ONE FOR THE STANDARD MODEL '''
    
    # Effective affinity
    KM_num = KD*(param.rD*param.vD*KD*(kp_const + param.rt*(1 + param.cDo/KD)) + kp_const*(param.rD + param.rt*(1 + 2*param.cDo/KD)))
    KM_denom = param.rt + 2*kp_const*param.rD*(1 + param.vD*KD)
    KM = KM_num/KM_denom

    # Reversibility factor
    rev = 1 - param.cDo/cDc

    # Prefactor
    pref = kp_const/(1 + 2*kp_const*(1 + param.vD*KD)/(param.rt*param.vD*KD))

    J = pref*rev*cDc/(cDc + KM)
    return J