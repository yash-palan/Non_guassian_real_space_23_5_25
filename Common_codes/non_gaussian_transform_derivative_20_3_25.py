# -*- coding: utf-8 -*-
"""
Created on  Sep 13, 2024

@author: Yash_palan

File containing the class definition for the input variables and the computed variables class

------------------------------------------
-----------------------------------------
"""
##############################################################################
##############################################################################
import numpy as np
import Common_codes.class_defn_file_20_3_25 as cdf
from Common_codes import correlation_functions_file_20_3_25 as cf
import torch

##############################################################################
##############################################################################
# Functions which compute the partial derivatives of the time evolution operator

def O_delta(time_derivative_lambda_bar:torch.Tensor,Gamma_m:torch.Tensor,input_variables:cdf.input_variables,
            correlation_matrices:cf.correlation_functions)->torch.Tensor:
    # Need to check this function once
    # Seems to work perfectly fine : Yash 10/5/24 17:06
    """
    This function computes the functional derivative with respect to bosonic averages (\\Delta_R) of the
    derivative of the Non Gaussian transformation (O = U^{-1}_S \\partial_t U_S).
    
    -----------------------------
    -----------------------------
    Parameters:
    -----------------------------
    -----------------------------
    
    time_derivative_lambda_bar:
    TYPE: complex torch tensor 
    SIZE: : (N_b, N_f)
    DESCRIPTION: This matrix stores the time derivative of the Non-Gaussian parameter and is to be used extracted from
    the odeint as well. However, doing this can be a little more tricky and needs to be monitored properly.

    Gamma_f:
    TYPE: complex torch tensor
    SIZE: N_f x N_f
    DESCRIPTION: This is the matrix that is used in the Hamiltonian.

    -----------------------------
    -----------------------------
    RETURNS
    -----------------------------
    -----------------------------

    final_mat_2:
    TYPE: complex numpy array
    SIZE: (2N_b,) 
    DESCRIPTION: This is the final matrix that is returned.

    """
    # volume = input_variables.volume
    N_f = input_variables.N_f
    N_b = input_variables.N_b

    final_mat = torch.zeros((2*N_b,),dtype = torch.complex128)
    c_dagger_c_mat = correlation_matrices.c_dagger_c_mat
    # temp_mat = 1j*np.einsum('kj,j->k',time_derivative_lambda_bar,(1+np.diag(Gamma_m[0:N_f,N_f:])))
    # temp_mat = 1j*torch.einsum('kj,j->k',time_derivative_lambda_bar,(1+torch.diag(Gamma_m[0:N_f,N_f:])))

    temp_mat = 2j*torch.einsum('kj,j->k',time_derivative_lambda_bar,torch.diag(c_dagger_c_mat))
    # final_mat =  np.append(np.zeros((N_b,)), temp_mat, axis=0)
    final_mat[N_b:] = temp_mat
    return(final_mat )

def O_b(time_derivative_lambda_bar:torch.Tensor,input_variables:cdf.input_variables,correlation_matrices:cf.correlation_functions)->torch.Tensor:
    # Dont think this one has any issues since this is effectively 0. Yash 17:06 10/5/24
    """
    This function computes the functional derivative with respect to bosonic correlation matrix ($\\Gamma_b$) of the
    derivative of the Non Gaussian transformation ($O = U^{-1}_S \\partial_t U_S$).
    
    However, for our case it is 0 and hence we do not necessarily need to give data to compute this.
    However, this is still in the code incase in the future we have a model for which this term is not 0.

    time_derivative_lambda_bar:
    TYPE: numpy array
    SIZE: 2N_b x N_f
    DESCRIPTION: This matrix stores the time derivative of the Non-Gaussian parameter and is to be used extracted from
    the odeint as well. However, doing this can be a little more tricky and needs to be monitored properly.

    Gamma_f:
    TYPE: numpy array
    SIZE: N_f x N_f
    DESCRIPTION: This is the matrix that is used in the Hamiltonian.

    -----------------------------
    -----------------------------
    RETURNS
    -----------------------------
    -----------------------------

    final_mat_2:
    TYPE: complex numpy array
    SIZE: 2N_b x 2N_b 
    DESCRIPTION: This is the final matrix that is returned.
    """
    N_b = input_variables.N_b
    final_array = torch.zeros((2*N_b,2*N_b)).to(dtype = torch.complex128)
    return final_array

def o_m(time_derivative_lambda_bar:torch.Tensor,Delta_R:torch.Tensor,input_variables:cdf.input_variables,correlation_matrices:cf.correlation_functions)->torch.Tensor:
    # Need to check this function once
    # Seems to be working fine : Yash 10/5/24 17:23
    """
    This function computes the 

    -----------------------------
    -----------------------------
    Parameters:
    -----------------------------
    -----------------------------
    time_derivative_lambda_bar: Type: numpy array
                                Size: 2*N_b x N_f
                                Description:
    
    Delta_R:    Type: numpy array
                Size: 1 x N_b
                Description: 
    -----------------------------
    -----------------------------
    RETRUNS
    -----------------------------
    -----------------------------
    final_array: TYPE: complex numpy array
         SIZE: N_f x N_f
         DESCRIPTION: This is the final matrix that is returned.
    """
    N_b = input_variables.N_b
    final_array = 1j*torch.diag(torch.einsum('ki,k->i',time_derivative_lambda_bar,Delta_R[N_b:2*N_b]))
    return final_array

def O_m(time_derivative_lambda_bar:torch.Tensor,Delta_R:torch.Tensor,input_variables:cdf.input_variables,correlation_matrices:cf.correlation_functions)->torch.Tensor:
    # Seems to be working fine : Yash 10/5/24 17:25
    """
    This function computes the partial derivatives of the time evolution operator with respect to the variation of omega(k) and gamma(k).

    Parameters:
    - time_derivative_lambda_bar: numpy array of shape (1, N_b)
        The variation of omega(k), the frequency of the bosons (omega), with momentum (k).
    - Delta_R: numpy array of shape (1, N_b)
        The variation of gamma(k), the electron-phonon interaction strength, with momentum (k).

    -----------------------------
    -----------------------------
    Parameters:
    -----------------------------
    -----------------------------
    time_derivative_lambda_bar: Type: numpy array
                                Size: 2*N_b x N_f
                                Description: array holding the time derivative of the Non-Gaussian parameter.
    
    Delta_R:    Type: numpy array
                Size: 1 x N_b
                Description: Expectation value of the Quadrature operators.
    -----------------------------
    -----------------------------
    Returns
    -----------------------------
    -----------------------------
    - final_array: complex numpy array of shape (2N_f, 2N_f)
        The partial derivatives of the time evolution operator.
    """
    N_f = input_variables.N_f

    o_m_temp_array = o_m(time_derivative_lambda_bar, Delta_R, input_variables,correlation_matrices)
    # mat_2 = [[0, 1], [-1, 0]]

    final_array = torch.zeros((2*N_f,2*N_f),dtype = torch.complex128)
    final_array[0:N_f,N_f:] = o_m_temp_array
    final_array[N_f:,0:N_f] = -o_m_temp_array
   
    # final_array = np.kron(mat_2, o_m_temp_array)    
    return (final_array)

##############################################################################
##############################################################################