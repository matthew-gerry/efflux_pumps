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

def efflux_numerical_3(param, KD, Kp, V_base, kappa, cDc, cpp, dchi):
    ''' GET MEAN EFFLUX RATE AND VARIANCE BY NUMERICALLY DIFFERENTIATING THE CGF, 3-STATE MODEL '''
    R = rm.rate_matrix_3(param, KD, Kp, V_base, kappa, cDc, cpp)

    # Find steady state solution as eigenvector of R associated with the zero eigenvalue
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] # Efflux at steady state

    # Do full counting statistics for the variance
    CGF = rm.cgf_3(R, dchi, 1) # Set chisteps to 1 since we just need the variance
    efflux_var = -np.diff(CGF, n=2)/(dchi**2) # Variance at steady state is given by the second derivative of the CGF wrt j*chi

    return efflux, efflux_var


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

    # Components of the MM-like expression
    Z = 1 + param.vD*param.vp*KD*Kp*KG # A partition function for states 1 and 3
    C1 = 1 + param.vD*KD*KG + param.vD*param.vp*KD*Kp*KG
    KM = (KD*(cpp*param.vp*KG + C1) + param.cDo*param.cpc*(param.vD*KD + param.vp*(Kp + cpp))/Kp)/(cpp*Z/Kp + C1)

    return KM


def spec_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    ''' SPECIFICITY OF THE PUMP (3-STATE MODEL) BASED ON MICHAELIS-MENTENT EXPRESSIONS '''
    
    KG = get_derived_params(param, cpp, V_base, kappa)[2]
    kt = param.r0*param.vD*param.vp*KD*Kp*KG/(1 + param.vD*param.vp*KD*Kp*KG) # Expression for the rotation transition rate

    Z = 1 + param.vD*param.vp*KD*Kp*KG # A partition function for states 1 and 3
    Kbeta = Kp*(1 + param.vD*KD*KG/Z) # Michaelis-Menten constants
    KM = KM_3(param, KD, Kp, V_base, kappa, cpp)

    alpha = 1 - param.cDo*param.cpc/(cDc*cpp*KG) # Efficiency factor (1 in the irreversible limit)

    S = alpha*kt*cpp/(KM*(Kbeta + cpp)) # Specificity as defined in original manuscript, with alpha folded in
    
    return S
    

def efflux_MM_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    ''' EFFLUX CALCULATED FROM THE MICHAELIS-MENTEN LIKE EXPRESSION DERIVED FOR THE REVERSIBLE CASE, THREE-STATE MODEL '''

    KM = KM_3(param, KD, Kp, V_base, kappa, cpp) # Michaelis-Menten constant
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

def efflux_numerical_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp, dchi):
    ''' GET MEAN EFFLUX RATE AND VARIANCE BY NUMERICALLY DIFFERENTIATING THE CGF, 8-STATE MODEL '''
    
    R = rm.rate_matrix_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp)

    # Find steady state solution as eigenvector of R
    SS = rm.steady_state(R)
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0] + SS[3]*R[4,3] - SS[4]*R[3,4] # Efflux at steady state

    # Do full counting statistics for the variance
    CGF = rm.cgf_8(R, dchi, 1) # Set chisteps to 1 since we just need the variance
    efflux_var = -np.diff(CGF, n=2)/(dchi**2) # Variance at steady state is given by the second derivative of the CGF wrt j*chi

    return efflux, efflux_var


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