# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:48:18 2023

@author: Yash_palan

This code contains the code that will
"""

#########################################
#########################################
from scipy.integrate import odeint
import numpy as np
from Common_codes.class_defn_file_18_10_24 import *
from Common_codes.generic_codes_18_10_24 import *
import scipy.linalg as spla
##############################################################################
##############################################################################
# Initialisation of the input matrices 

def initialise_delta_R_matrix(N_b:int,seed:float)->np.ndarray:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    Function to compute a random matrix used for the initialisation of the variational parameter Delta_R

    -----------------------------
    -----------------------------
    Parameters:
    -----------------------------
    -----------------------------
    seed:   Type: float
            Description: This is the seed used for the random number generator.

    N_b:    Type: int
            Description: This is the number of bosons in the system.
    -----------------------------
    -----------------------------
    Returns:
    -----------------------------
    -----------------------------
    Type: complex numpy array
    Shape: (2*N_b)     
    """
    # Set the seed
    # np.random.seed(seed)
    return(np.random.rand(2*N_b))

def initialise_Gamma_b_matrix(N_b:int,seed:float)->np.ndarray:

    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    Function to compute a random matrix used for the initialisation of the variational parameter Gamma_b

    -----------------------------
    -----------------------------
    Parameters:
    -----------------------------
    -----------------------------
    seed:   Type: float
            Description: This is the seed used for the random number generator.

    N_b:    Type: int
            Description: This is the number of bosons in the system.
    -----------------------------
    -----------------------------
    Returns: Gamma_b
    -----------------------------
    -----------------------------
    Type: complex numpy array
    Shape: (2*N_b,2*N_b) 
    
    """
    # Set the seed
    # np.random.seed(seed)

    # Creating a random Real Symmetric matrix (not necessarily Haar Distributed) 
    xi_b = np.random.rand(2*N_b,2*N_b)
    xi_b = 0.5*(xi_b + xi_b.T)
    # xi_b = 0.5/(N_b)*(xi_b + xi_b.T)

    # Creating the corresponding Symplectic matrix  
    sigma_mat = sigma(N_b,dtype="float")
    S_b = spla.expm(np.matmul(sigma_mat,xi_b))

    # Creating the corresponding COvariance matrix
    Gamma_b = np.matmul(S_b,S_b.T)

    if(True in set(np.reshape( np.imag(Gamma_b)> 1e-10 ,(-1)) ) ):
        raise Exception("Imaginary part of the Matrix is not zero. Please check the computation of the matrix.")

    return(Gamma_b)

def initialise_Gamma_m_matrix(N_f:int,seed:float)->np.ndarray:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    Function to compute a random matrix used for the initialisation of the variational parameter Gamma_f

    -----------------------------
    -----------------------------
    Parameters:
    -----------------------------
    -----------------------------
    seed:   Type: float
            Description: This is the seed used for the random number generator.

    N_f:    Type: int
            Description: This is the number of Fermions in the system.
    -----------------------------
    -----------------------------
    Returns: Gamma_m
    -----------------------------
    -----------------------------
    Type: complex numpy array
    Shape: (2*N_f,2*N_f) 
    """
    # Note: This may see some change as the I am not sure of the defintion of the function xi_m since
    # it has to be completely antisymmetric and hermitian (which also seems to indicate that it is completely
    # imaginary).

    # Set the seed
    # np.random.seed(seed)

    # Creating a random Real Anti-Symmetric,Hermitian matrix (not necessarily Haar Distributed)
    # The anti-symmetry and Hermitian implies that the matrix is completely imaginary 
    xi_m = np.random.rand(2*N_f,2*N_f)
    xi_m = 0.5*1j*(xi_m - xi_m.T)

    # Creating the corresponding Orthogonal matrix 
    U_m = spla.expm(1j*xi_m)

    sigma_mat = sigma(N_f,dtype="float")
    
    # Creating the corresponding COvariance matrix
    Gamma_m = -np.matmul(U_m,np.matmul(sigma_mat,U_m.T))

    if(True in set(np.reshape( np.imag(Gamma_m)> 1e-10 ,(-1)) ) ):
        raise Exception("Imaginary part of the Matrix is not zero. Please check the computation of the matrix.")

    return(Gamma_m)

