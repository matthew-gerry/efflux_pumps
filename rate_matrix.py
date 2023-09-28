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
    
    CALLS A Params3 OBJECT AS DEFINED IN params.py
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


#### FUNCTIONS: EIGHT-STATE MODEL ####

def rate_matrix_8(param, KD_list, Kp_list, V_base, kappa, cDc, cpp):
    '''
    RATE MATRIX FOR THE EIGHT-STATE KINETIC MODEL

    CALLS A Params8 OBJECT AS DEFINED IN params.py
    '''
    
    # Electric potential Boltzmann factor
    KG = get_derived_params(param, cpp, V_base, kappa)[2]

    ### FORWARD RATE CONSTANTS ###
    # Forward rate constants, cycle A (one drug, one proton)
    kDA = param.rD*param.vD_list[0] # Drug binding, cycle A
    kpA = param.rp*param.vp_list[0] # Proton binding, cycle A
    ktA = param.rt*param.vD_list[0]*param.vp_list[0]*Kp_list[0]*KD_list[0]*KG/(1 + param.vD_list[0]*param.vp_list[0]*Kp_list[0]*KD_list[0]*KG) # Rotation, cycle A

    # Forward rate constants, cycle B (one drug, two protons)
    kpB1 = param.rp*param.vp_list[1] # Proton binding (first), cycle B
    ktB1 = param.rt*param.vD_list[1]*param.vp_list[1]*Kp_list[1]*KD_list[1]*KG/(1 + param.vD_list[1]*param.vp_list[1]*Kp_list[1]*KD_list[1]*KG) # Rotation (first), cycle B
    kDB = param.rD*param.vD_list[1] # Drug binding, cycle B
    kpB2 = param.rp*param.vp_list[2] # Proton binding (second), cycle B
    ktB2 = param.rt*param.vp_list[2]*Kp_list[2]*KG/(1 + param.vp_list[2]*Kp_list[2]*KG) # Rotation (first), cycle B

    # Forward rate constants, cycle B (zero drugs, one proton)
    kpC = param.rp*param.vp_list[3] # Proton binding, cycle C
    ktC = param.rt*param.vp_list[3]*Kp_list[3]*KG/(1 + param.vp_list[3]*Kp_list[3]*KG) # Rotation, cycle C

    ### RATE MATRIX ###
    R = np.zeros([8,8]) # Initialize rate matrix
    # Insert nonzero transition rates to off-diagonal elements
    R[0,1] = kDA*KD_list[0]; R[0,2] = ktA; R[0,3] = kpB1*Kp_list[1]; R[0,7] = ktB2
    R[1,0] = kDA*cDc; R[1,2] = kpA*Kp_list[0]
    R[2,0] = ktA*param.cDo*param.cpc/(KD_list[0]*Kp_list[0]*KG); R[2,1] = kpA*cpp

    R[3,0] = kpB1*cpp; R[3,4] = ktB1*param.cDo*param.cpc/(KD_list[1]*Kp_list[1]*KG)

    R[4,3] = ktB1; R[4,5] = kpC*Kp_list[3] + ktC; R[4,6] = kDB*KD_list[1]
    R[5,4] = kpC*cpp + ktC*param.cpc/(Kp_list[3]*KG)

    R[6,4] = kDB*cDc; R[6,7] = kpB2*Kp_list[2]
    R[7,6] = kpB2*cpp; R[7,0] = ktB2*param.cpc/(Kp_list[2]*KG)
    
    # Get diagonal elements from normalization condition
    for i in range(8):
        R[i,i] = -sum(R)[i]

    return R

def cgf_8(R, dchi, chisteps):
    ''' CUMULANT GENERATING FUNCTION FOR THE EIGHT-STATE MODEL, GIVEN A RATE MATRIX AND PARAMETERS DESCRIBING THE CHI AXIS '''

    # chisteps - number of steps along the chi axis from zero to the max value
    # Define the chi axis such that it is symmetric about zero
    chiplus = np.linspace(0,dchi*chisteps,chisteps+1)
    chi_axis = np.concatenate([-np.flip(chiplus)[:-1],chiplus])

    CGF = np.zeros(len(chi_axis)) # Allocate list to hold CGF values
    for i,chi in enumerate(chi_axis):
        R_chi = R
        R_chi[0,2] = R[0,2]*np.exp(complex(0,chi)) # Dress the generator with a counting field
        R_chi[2,0] = R[2,0]*np.exp(complex(0,-chi)) # Vanishes with the irreversibility assumption
        R_chi[4,3] = R[4,3]*np.exp(complex(0,chi))
        R_chi[3,4] = R[3,4]*np.exp(complex(0,-chi))


       # The CGF is the eigenvalue whose real part approaches zero as chi -> 0
        eig_chi = np.linalg.eig(R_chi)[0]
        CGF[i] = eig_chi[eig_chi.real==max(eig_chi.real)]

    return CGF


#### FUNCTIONS: FIVE-STATE MODEL ####


def rate_matrix_5(param, KD, Kp, QD, Qp, V_base, kappa, cDc, cpp):
    '''
    RATE MATRIX FOR THE EFFLUX PUMP, FIVE-STATE KINETIC MODEL
    
    CALLS A Params3 OBJECT AS DEFINED IN params.py
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

#### FUNCTIONS: FOUR-STATE MODEL ####

def rate_matrix_4(param, KD, Kp_list, V_base, kappa, cDc, cpp):
    '''
    RATE MATRIX FOR THE FOUR-STATE KINETIC MODEL

    CALLS A Params4 OBJECT AS DEFINED IN params.py
    '''
    
    # Electric potential Boltzmann factor
    KG = get_derived_params(param, cpp, V_base, kappa)[2]

    ### FORWARD RATE CONSTANTS ###
    # Forward rate constants, cycle A (one drug, one proton)
    kDA = param.rD*param.vD # Drug binding, pumping cycle
    kpA = param.rp*param.vp_list[0] # Proton binding, pumping cycle
    ktA = param.rt*param.vD*param.vp_list[0]*KD*Kp_list[0]*KG/(1 + param.vD*param.vp_list[0]*KD*Kp_list[0]*KG) # Conformational transition, pumping cycle

    # Forward rate constants, cycle B (zero drugs, one proton)
    kpB = param.rp*param.vp_list[1] # Proton binding, waste cycle
    ktB = param.rt*param.vp_list[1]*Kp_list[1]*KG/(1 + param.vp_list[1]*Kp_list[1]*KG) # Conformational transition, waste cycle

    ### RATE MATRIX ###
    R = np.zeros([4,4]) # Initialize rate matrix
    # Insert nonzero transition rates to off-diagonal elements
    R[0,1] = kDA*KD; R[0,2] = ktA; R[0,3] = kpB*Kp_list[1] + ktB
    R[1,0] = kDA*cDc; R[1,2] = kpA*Kp_list[0]
    R[2,0] = ktA*param.cDo*param.cpc/(KD*Kp_list[0]*KG); R[2,1] = kpA*cpp
    R[3,0] = kpB*cpp + ktB*param.cpc/(Kp_list[1]*KG)
    
    # Get diagonal elements from normalization condition
    for i in range(4):
        R[i,i] = -sum(R)[i]

    return R

#### FUNCTIONS: SEVEN-STATE MODEL ####

def rate_matrix_7(param, KD, Kp_list, QD, Qp_list, V_base, kappa, cDc, cpp):
    '''
    RATE MATRIX FOR THE EFFLUX PUMP, SEVEN-STATE KINETIC MODEL
    
    CALLS A Params3 OBJECT AS DEFINED IN params.py
    '''

    # Electric potential Boltzmann factor
    KG = get_derived_params(param, cpp, V_base, kappa)[2]

    Kp_pump = Kp_list[0]; Kp_waste = Kp_list[1]
    Qp_pump = Qp_list[0]; Qp_waste = Qp_list[1]

    # Forward rate constants
    kD = param.rD*param.vD # Drug binding
    kp = param.rp*param.vp # Proton binding
    kt_pump = param.rt/(1 + QD*Qp_pump/(KG*KD*Kp_pump))
    kt_waste = param.rt/(1 + Qp_waste/(KG*Kp_waste))

    R = np.zeros([7,7]) # Initialize rate matrix
    # Insert transition rates related to pump cycle...
    R[0,1] = kD*KD; R[0,4] = kD*QD
    R[1,0] = kD*cDc; R[1,2] = kp*Kp_pump
    R[2,1] = kp*cpp; R[2,3] = kt_pump*QD*Qp_pump/(KG*KD*Kp_pump)
    R[3,2] = kt_pump; R[3,4] = kp*param.cpc
    R[4,3] = kp*Qp_pump; R[4,0] = kD*param.cDo

    # ... and waste cycle
    R[0,5] = kp*Kp_waste; R[0,6] = kp*Qp_waste
    R[5,0] = kp*cpp; R[5,6] = kt_waste*Qp_waste/(KG*Kp_waste)
    R[6,5] = kt_waste; R[6,0] = kp*param.cpc

    # Get diagonal elements from normalization condition
    for i in range(7):
        R[i,i] = -sum(R)[i]

    return R

#### FUNCTIONS: GENERAL ####

def steady_state(R):
    ''' GIVEN A STOCHASTIC RATE MATRIX, COMPUTES THE STEADY STATE POPULATIONS '''
    # Will output garbage if the input is not a valid rate matrix

    eigvals_vecs = np.linalg.eig(R)
    SS_unnormalized = eigvals_vecs[1][:,np.real(eigvals_vecs[0])==max(np.real(eigvals_vecs[0]))]
    SS = np.real(SS_unnormalized)/sum(np.real(SS_unnormalized)) # Normalize

    return SS