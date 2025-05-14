# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:41:53 2023

@author: Yash_palan
"""
############################################
############################################
import numpy as np
############################################
############################################
def sigma(N_b,dtype="float"):
    # Check if this gives the correct ouput
    # Gives the correct output : Yash 6/10/2023 12:04
    """
    This function creates the sigma matrix. 
    The sigma matrix is the matrix which is used in the definition of a Symplectic matrix.

    Parameters:

    N_b:    TYPE: int
            SIZE: 1
            DESCRIPTION: Number of bosons, or basically the number of momentum points in the k grid 
                        (as the number of bosons is given by the discretisation of the bosonic annihilation 
                        operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
                        momentum grid).

    Returns: 
    Size: 2N_b x 2N_b
    Desceiption: Returns the Symplectic sigma/Omega matrix (which is 2N_b x 2N_b matrix)
    
    """
    mat_1 = np.array([[0.0,1.0],[-1.0,0.0]])
    mat_2 = np.identity(N_b)
    final_mat=np.kron(mat_1, mat_2)
    final_mat = final_mat.astype(dtype)
    return(final_mat)
############################################
############################################
def check_majorana_covariance_matrix(Gamma_m:np.ndarray):
    """
    This function just checks some properties of the Majorana covariance matrix to ascetain
    that the program does what it has to do correctly
    ----------
    Parameters
    ----------
    Gamma_f :   Size: (2N_f,2N_f) real numpy array
                Description: The bosonic quadrature covariance matrix
    """

    if(True in set( np.abs( np.reshape( np.matmul(Gamma_m,Gamma_m) + np.eye(len(Gamma_m)),(-1)) ) > 1e-4 ) ):
        raise Exception("The condition for a Pure Fermionic state is not satisfied. Exiting the program.")
    if(True in set( np.abs( np.reshape( Gamma_m.T + Gamma_m ,(-1)) ) > 1e-10 ) ):
        raise Exception("The condition for the Majorana covariance matrix being Anti-Symmetric is not satisfied. Exiting the program.")
    if(True in set( np.reshape( np.imag( Gamma_m ),(-1)) > 1e-10 ) ):
        raise Exception("The condition for the Majorana covariance matrix being Real is not satisfied. Exiting the program.")
    return
############################################
############################################
def check_bosonic_quadrature_covariance_matrix(Gamma_b:np.ndarray):
    """
    This function just checks some properties of the bosonic quadrature covariance matrix to ascetain
    that the program does what it has to do correctly
    ----------
    Parameters
    ----------
    Gamma_b :   Size: (2N_b,2N_b) real numpy array
                Description: The bosonic quadrature covariance matrix
    """

    N_b = int(Gamma_b.shape[0]/2)
    sigma_mat = sigma(N_b)

    if(True in set( np.abs( np.reshape(  np.matmul(Gamma_b,np.matmul(sigma_mat,Gamma_b.T)) - sigma_mat ,(-1)) ) > 1e-4 ) ):
        raise Exception("The condition for the quadrature covariance matrix to be Symplectic is not satisfied. Exiting the program.")
    if(True in set( np.abs( np.reshape( Gamma_b.T - Gamma_b ,(-1)) ) > 1e-10 ) ):
        raise Exception("The condition for the bosonic quadrature covariance matrix being Symmetric is not satisfied. Exiting the program.")
    if(True in set( np.reshape(np.abs( np.imag( Gamma_b )) , (-1)  ) > 1e-10 ) ):
        raise Exception("The condition for the bosonic quadrature covariance matrix being Real is not satisfied. Exiting the program.")
    return
############################################
############################################
def check_bosonic_quadrature_average_matrix(Delta_r:np.ndarray):
    """
    This function just checks some properties of the bosonic quadrature average matrix to ascetain
    that the program does computations correctly.

    ----------
    Parameters
    ----------
    Delta_r :   Size: (2N_b,) real numpy array
                Description: The bosonic quadrature average matrix
    """
    if(True in set( np.reshape( np.imag( Delta_r ),(-1)) > 1e-10 ) ):
        raise Exception("The condition for the bosonic_quadrature_average covariance matrix being Real is not satisfied. Exiting the program.")
    return
############################################
############################################