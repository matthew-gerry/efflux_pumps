'''
efflux_pumps.py

Functions compute the efflux and related quantities for bacterial efflux pumps
using the three- and eight-state model

Matthew Gerry, March 2023
'''

from parameters import *
import rate_matrix as rm

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

def efflux_numerical_3(param, KD, Kp, V_base, kappa, cDc, cpp, dchi):
    ''' GET MEAN EFFLUX RATE AND VARIANCE BY NUMERICALLY DIFFERENTIATING THE CGF '''
    R = rm.rate_matrix_3(param, KD, Kp, V_base, kappa, cDc, cpp)

    # Find steady state solution as eigenvector of R associated with the zero eigenvalue
    eigvals_vecs = np.linalg.eig(R)
    SS_unnormalized = eigvals_vecs[1][:,np.real(eigvals_vecs[0])==max(np.real(eigvals_vecs[0]))]
    # global SS
    SS = np.real(SS_unnormalized)/sum(np.real(SS_unnormalized)) # Normalize

    # Efflux at steady state
    efflux = SS[2]*R[0,2] - SS[0]*R[2,0]

    # Do full counting statistics
    CGF = rm.cgf_3(R, dchi, 1) # Set chisteps to 1 since we just need the variance
    efflux_var = -np.diff(CGF, n=2)/(dchi**2) # Variance at steady state is given by the second derivative of the CGF wrt j*chi

    return efflux, efflux_var

# ADD SEPARATE FUNCTION FOR KM, SPECIFICITY TO THEN BE CALLED IN THE BODY OF THE MM FUNCTION

def efflux_MM_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    ''' EFFLUX CALCULATED FROM THE MICHAELIS-MENTEN LIKE EXPRESSION DERIVED FOR THE REVERSIBLE CASE, THREE-STATE MODEL '''

    KG = get_derived_params(param, cpp, V_base, kappa)[2]
    kt = r0*vD*vp*KD*Kp*KG/(1 + vD*vp*KD*Kp*KG) # Expression for the rotation transition rate

    # Components of the MM-like expression
    Z = 1 + param.vD*param.vp*KD*Kp*KG # A partition function for states 1 and 3
    C1 = 1 + param.vD*KD*KG + param.vD*param.vp*KD*Kp*KG
    Kbeta = Kp*(1 + param.vD*KD*KG/Z) # Michaelis-Menten constants
    KM = (KD*(cpp*param.vp*KG + C1) + param.cDo*param.cpc*(param.vD*KD + param.vp*(Kp + cpp))/Kp)/(cpp*Z/Kp + C1)

    alpha = 1 - param.cDo*param.cpc/(cDc*cpp*KG) # Efficiency factor

    # num = kt*cDc*cpp - kt*cDo*cpc/KG # Efflux expression
    num = alpha*kt*cDc*cpp
    denom = (KM + cDc)*(Kbeta + cpp)

    efflux = num/denom # Efflux