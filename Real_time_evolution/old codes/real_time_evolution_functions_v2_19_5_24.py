# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:43:50 2023

@author: Yash_palan

This file is used for definining all the functions that we will need for Imaginary time evolution
of the state.

Latest update: 6/2/25 - Added the function for the equation of motion for the  

"""
##############################################################################
##############################################################################
import numpy as np
from scipy import linalg as la
import torch
from Common_codes import hamiltonian_derivative_matrices_18_10_24 as hdm
from Common_codes import non_gaussian_transform_derivative_18_10_24 as ngtd
from Common_codes import class_defn_file_18_10_24 as cdf
from Common_codes import correlation_functions_file_18_10_24 as cf
##############################################################################
##############################################################################
# First equation of motion functions

def equation_of_motion_for_Non_Gaussian_parameter_lambda(delta_r:np.ndarray,Gamma_b:np.ndarray,Gamma_m:np.ndarray,
                                                input_variables:cdf.input_variables,computed_variables:cdf.computed_variables):
    """
    Calculates the right-hand side (RHS) of the equation of motion for the non-Gaussian parameter.
    \\partial_{\\tau} \\lambda_q is computed rather than  \\partial_{\\tau} \\lambda_bar
    
    
    Parameters:
    ------------

    Returns:
    ------------
    time_derivative_lambda_bar : 
    TYPE: complex numpy array
    SIZE: 1 x N_b 
    Matrix which stores the time derivative of the non-Gaussian parameter.
    """
    N_f = input_variables.N_f

    correlation_mat_for_non_gaussian_parameters = cf.correlation_matrix_creation(Gamma_m,N_f)
    denominator = np.einsum('qm,mn->q',np.conj(input_variables.fourier_array),correlation_mat_for_non_gaussian_parameters)

    kappa_1_new_mat = hdm.rhs_var_par_eqn_of_motion_lambda_real_time(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,
                                                        correlation_mat_for_non_gaussian_parameters)

    time_derivative_lambda_bar = kappa_1_new_mat/denominator
    return(time_derivative_lambda_bar)

def equation_of_motion_for_bosonic_averages(delta_r:np.ndarray,Gamma_b:np.ndarray,Gamma_m:np.ndarray,time_derivative_lambda_bar,
                                            input_variables,computed_variables):

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 
    Here we also calculate the contribution of the bosonic average derivatives to the phase term (i.e.propto h_{\\delta} and O_{\\delta}).

    Parameters
    ----------
    
    delta_R : TYPE:
              SIZE:
              DESCRIPTION.
              
    Gamma_b : TYPE:
                  
              DESCRIPTION.
              
    Gamma_f : TYPE:
              SIZE:
              DESCRIPTION.

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 1 x 2N_b
    """
    N_b = input_variables.N_b
    sigma = torch.from_numpy(np.kron([[0,1],[-1,0]],np.eye(N_b))).to(torch.complex128)
    h_delta_matrix = hdm.h_delta(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)
    O_delta_mat = ngtd.O_delta(time_derivative_lambda_bar,Gamma_m,input_variables)
    # sigma = np.kron([[0,1],[-1,0]],np.eye(N_b))

    # h_delta_matrix = gf.h_delta(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)
    # O_delta_mat = gf.O_delta(time_derivative_lambda_bar,Gamma_m,input_variables)
    delta_r_tensor = torch.tensor(delta_r,dtype=torch.complex128)

    final_mat = torch.matmul(sigma, h_delta_matrix) - complex(0,1)*torch.matmul(sigma,O_delta_mat)
    phase_term = (1/4.0)*torch.matmul(delta_r_tensor.T,h_delta_matrix-complex(0,1)*O_delta_mat)
    return([final_mat,phase_term])

def equation_of_motion_for_bosonic_covariance(delta_r,Gamma_b,Gamma_m,time_derivative_lambda_bar,
                                                input_variables,computed_variables):

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 
    Here we also calculate the contribution of the O_b terms to the computation of the phase term.

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 2N_b x 2N_b
    """

    N_b = input_variables.N_b

    sigma = torch.tensor(np.kron([[0,1],[-1,0]],np.eye(N_b))  ,dtype=torch.complex128)
    Gamma_b_tensor = torch.tensor(Gamma_b,dtype=torch.complex128)

    # Remember that these definions are extremely important for the calculation of the h_b matrix and should
    # be passed correctly to the function. Any change in the definition of the h_b function SHOULD be reflected
    # in here as well

    h_b_matrix = hdm.h_b(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)

    # Remember that these definions are extremely important for the calculation of the O_b matrix and should
    # be passed correctly to the function. Any change in the definition of the O_b function SHOULD be reflected
    # in here as well
    O_b_matrix = ngtd.O_b(time_derivative_lambda_bar,input_variables)

    final_mat = (torch.matmul(sigma,torch.matmul(h_b_matrix-complex(0,1)*O_b_matrix,Gamma_b_tensor)) \
                - torch.matmul( Gamma_b_tensor,torch.matmul(h_b_matrix-complex(0,1)*O_b_matrix,sigma)) )
    
    # final_mat = (np.matmul(sigma,np.matmul(h_b_matrix-complex(0,1)*O_b_matrix,Gamma_b)) \
    #                 -np.matmul(Gamma_b,np.matmul(h_b_matrix-complex(0,1)*O_b_matrix,sigma)) \
    #                 )
    phase_term = 1/4.0*torch.trace(O_b_matrix) -1j/4.0*torch.trace( Gamma_b_tensor @ sigma @ O_b_matrix )
    return([final_mat,phase_term])

def equation_of_motion_for_fermionic_covariance(delta_r,Gamma_b,Gamma_m,time_derivative_lambda_bar,
                                                input_variables,computed_variables):

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 
    Here we also calculate the contribution of the O_m terms to the computation of the phase term.

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 2N_b x 2N_b
    
    """
    h_m_matrix = hdm.h_m(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)
    O_m_matrix = ngtd.O_m(time_derivative_lambda_bar,delta_r,input_variables)
    Gamma_m_tensor = torch.tensor(Gamma_m,dtype=torch.complex128)

    final_mat = (torch.matmul(h_m_matrix-complex(0,1)*O_m_matrix, Gamma_m_tensor)  
                - torch.matmul( Gamma_m_tensor,h_m_matrix-complex(0,1)*O_m_matrix) 
                )

    # final_mat = ( np.matmul(h_m_matrix-complex(0,1)*O_m_matrix, Gamma_m) \
    #              -np.matmul( Gamma_m,h_m_matrix-complex(0,1)*O_m_matrix)
    #             )
    phase_term = 1j/4.0*torch.trace(O_m_matrix) - 1/4.0*torch.trace(Gamma_m_tensor @ O_m_matrix) 
    return([final_mat,phase_term])

##############################################################################
##############################################################################