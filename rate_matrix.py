'''
rate_matrix.py

Define the rate matrix for the efflux pump kinetic model (three- and eight-state).

Matthew Gerry, March 2023
'''

from parameters import *


#### FUNCTIONS: THREE-STATE MODEL ####

def rate_matrix_3(param, KD, Kp, V_base, kappa, cDc, cpp):
    '''
    RATE MATRIX FOR THE EFFLUX PUMP, THREE-STATE KINETIC MODEL
    
    CALLS A Parameters OBJECT AS DEFINED IN params.py
    '''

    # Electric potential Boltzmann factor
    KG = get_derived_params(param, cpp, V_base, kappa)[2]

    # Forward rate constants
    kD = param.rD*param.vD # Drug binding
    kp = param.rp*param.vp # Proton binding
    kt = param.rt*param.vD*param.vp*KD*Kp*KG/(1 + param.vD*param.vp*KD*Kp*KG)

    R = np.zeros([3,3]) # Initialize rate matrix
    # Insert transition rates
    R[0,1] = kD*KD; R[0,2] = kt
    R[1,0] = kD*cDc; R[1,2] = kp*Kp
    R[2,0] = kt*param.cDo*param.cpc/(KD*Kp*KG); R[2,1] = kp*cpp

    # Get diagonal elements from normalization condition
    for i in range(3):
        R[i,i] = -sum(R)[i]

    return R

def cgf_3(R, dchi, chisteps):
    ''' CUMULANT GENERATING FUNCTION, GIVEN A RATE MATRIX AND PARAMETERS DESCRIBING THE CHI AXIS '''

    # chisteps - number of steps along the chi axis from zero to the max value
    # Define the chi axis such that it is symmetric about zero
    chiplus = np.linspace(0,dchi*chisteps,chisteps+1)
    chi_axis = np.concatenate([-np.flip(chiplus)[:-1],chiplus])

    CGF = np.zeros(len(chi_axis)) # Allocate list to hold CGF values
    for i,chi in enumerate(chi_axis):
        R_chi = R
        R_chi[0,2] = R[0,2]*np.exp(complex(0,chi)) # Dress the generator with a counting field
        R_chi[2,0] = R[2,0]*np.exp(complex(0,-chi)) # Vanishes with the irreversibility assumption

       # The CGF is the eigenvalue whose real part approaches zero as chi -> 0
        eig_chi = np.linalg.eig(R_chi)[0]
        CGF[i] = eig_chi[eig_chi.real==max(eig_chi.real)]

    return CGF


#### FUNCTIONS: FIVE-STATE MODEL ####


def rate_matrix_5(param, KD, Kp, QD, Qp, V_base, kappa, cDc, cpp):
    '''
    RATE MATRIX FOR THE EFFLUX PUMP, FIVE-STATE KINETIC MODEL
    
    CALLS A Parameters OBJECT AS DEFINED IN parameters.py
    '''

    # Electric potential Boltzmann factor
    KG = get_derived_params(param, cpp, V_base, kappa)[2]

    # Forward rate constants
    kD = param.rD*param.vD # Drug binding
    kp = param.rp*param.vp # Proton binding
    kt = param.rt/(1 + QD*Qp/(KG*KD*Kp))

    R = np.zeros([5,5]) # Initialize rate matrix
    # Insert transition rates
    R[0,1] = kD*KD; R[0,4] = kD*QD
    R[1,0] = kD*cDc; R[1,2] = kp*Kp
    R[2,1] = kp*cpp; R[2,3] = kt*QD*Qp/(KG*KD*Kp)
    R[3,2] = kt; R[3,4] = kp*param.cpc
    R[4,3] = kp*Qp; R[4,0] = kD*param.cDo

    # Get diagonal elements from normalization condition
    for i in range(5):
        R[i,i] = -sum(R)[i]

    return R

def rate_matrix_5a(param, KD, Kp, QD, Qp, V_base, kappa, cDc, cpp):
    '''
    RATE MATRIX FOR FIVE-STATE CYCLE, REVERSED ORDER OF UNBINDING
    '''

    # Electric potential Boltzmann factor
    KG = get_derived_params(param, cpp, V_base, kappa)[2]

    # Forward rate constants
    kD = param.rD*param.vD # Drug binding
    kp = param.rp*param.vp # Proton binding
    kt = param.rt/(1 + QD*Qp/(KG*KD*Kp))

    R = np.zeros([5,5]) # Initialize rate matrix
    # Insert transition rates
    R[0,1] = kD*KD; R[4,3] = kD*QD
    R[1,0] = kD*cDc; R[1,2] = kp*Kp
    R[2,1] = kp*cpp; R[2,3] = kt*QD*Qp/(KG*KD*Kp)
    R[3,2] = kt; R[4,0] = kp*param.cpc
    R[0,4] = kp*Qp; R[3,4] = kD*param.cDo

    # Get diagonal elements from normalization condition
    for i in range(5):
        R[i,i] = -sum(R)[i]

    return R


def cgf_5(R, dchi, chisteps):
    ''' CUMULANT GENERATING FUNCTION FOR THE FIVE-STATE MODEL, GIVEN A RATE MATRIX AND PARAMETERS DESCRIBING THE CHI AXIS '''

    # chisteps - number of steps along the chi axis from zero to the max value
    # Define the chi axis such that it is symmetric about zero
    chiplus = np.linspace(0,dchi*chisteps,chisteps+1)
    chi_axis = np.concatenate([-np.flip(chiplus)[:-1],chiplus])

    CGF = np.zeros(len(chi_axis)) # Allocate list to hold CGF values
    for i,chi in enumerate(chi_axis):
        R_chi = R
        R_chi[2,3] = R[2,3]*np.exp(complex(0,chi)) # Dress the generator with a counting field
        R_chi[3,2] = R[3,2]*np.exp(complex(0,-chi)) # Vanishes with the irreversibility assumption

       # The CGF is the eigenvalue whose real part approaches zero as chi -> 0
        eig_chi = np.linalg.eig(R_chi)[0]
        CGF[i] = eig_chi[eig_chi.real==max(eig_chi.real)]

    return CGF


#### FUNCTIONS: SEVEN-STATE MODEL ####

def rate_matrix_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp):
    '''
    RATE MATRIX FOR THE EFFLUX PUMP, SEVEN-STATE KINETIC MODEL

    IN THIS MODEL, THE DRUG UNBINDS TO THE OUTSIDE, THEN THE PROTON TO THE CYTOPLASM
    
    CALLS A Parameters OBJECT AS DEFINED IN params.py
    '''

    # Electric potential Boltzmann factor
    KG = get_derived_params(param, cpp, V_base, kappa)[2]

    Kp_pump = Kp_list[0]
    Kp_waste = Kp_list[1]
    Qp_pump = Qp_list[0]
    Qp_waste = Qp_list[1]

    # Forward rate constants
    kD = param.rD*param.vD # Drug binding
    kp = param.rp*param.vp # Proton binding
    kt_pump = param.rt/(1 + QD*Qp_pump/(KG*KD*Kp_pump))
    kt_waste = param.rt/(1 + Qp_waste/(KG*Kp_waste))

    R = np.zeros([7,7]) # Initialize rate matrix
    # Insert transition rates related to pump cycle...
    R[0,1] = kD*KD; R[4,3] = kD*QD
    R[1,0] = kD*cDc; R[1,2] = kp*Kp_pump
    R[2,1] = kp*cpp; R[2,3] = kt_pump*QD*Qp_pump/(KG*KD*Kp_pump)
    R[3,2] = kt_pump; R[4,0] = kp*param.cpc
    R[0,4] = kp*Qp_pump; R[3,4] = kD*param.cDo

    # ... and waste cycle
    R[0,5] = kp*Kp_waste; R[0,6] = kp*Qp_waste
    R[5,0] = kp*cpp; R[5,6] = kt_waste*Qp_waste/(KG*Kp_waste)
    R[6,5] = kt_waste; R[6,0] = kp*param.cpc

    # Get diagonal elements from normalization condition
    for i in range(7):
        R[i,i] = -sum(R)[i]

    return R


#### FUNCTIONS: PROTON-INDEPENDENT ####

# def rate_matrix_p_ind(param, KD, cDc, kp_const):
#     ''' THREE-STATE MODEL REPLACING EACH PROTON-DEPENDENT STEP WITH A PROTON-INDEPENDENT ONE '''

#     # Set outside drug concentration cDo as a variable (override param setting)

#     # Forward rate constants
#     kD = param.rD*param.vD # Drug binding - same as standard model
#     kt = param.rt*param.vD*KD/(1 + param.vD*KD) # Multistep transition

#     R = np.zeros([3,3]) # Initialize rate matrix
#     # Insert transition rates
#     R[0,1] = kD*KD; R[0,2] = kt
#     R[1,0] = kD*cDc; R[1,2] = kp_const
#     R[2,0] = kt*param.cDo/KD; R[2,1] = kp_const

#     # Get diagonal elements from normalization condition
#     for i in range(3):
#         R[i,i] = -sum(R)[i]

#     return R


#### FUNCTIONS: GENERAL ####

def steady_state(R):
    ''' GIVEN A STOCHASTIC RATE MATRIX, COMPUTES THE STEADY STATE POPULATIONS '''
    # Will output garbage if the input is not a valid rate matrix

    eigvals_vecs = np.linalg.eig(R)
    SS_unnormalized = eigvals_vecs[1][:,np.real(eigvals_vecs[0])==max(np.real(eigvals_vecs[0]))]
    SS = np.real(SS_unnormalized)/sum(np.real(SS_unnormalized)) # Normalize

    return SS