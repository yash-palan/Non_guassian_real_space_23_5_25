# -*- coding: utf-8 -*-
"""
Created on March 25 2025
@author: Yash_palan

This file is used for definining all the functions that we will need for Imaginary time evolution
of the state.

Latest update: 6/2/25 - Added the function for the equation of motion for the  

"""
##############################################################################
##############################################################################
import time
import numpy as np
import torch
from Common_codes import hamiltonian_derivative_matrices_20_3_25 as hdm
from Common_codes import non_gaussian_transform_derivative_20_3_25 as ngtd
from Common_codes import class_defn_file_20_3_25 as cdf
from Common_codes import correlation_functions_file_20_3_25 as cf
# import h5py
##############################################################################
##############################################################################
# First equation of motion functions
# Function to create an HDF5 file and initialize a dataset


def equation_of_motion_for_Non_Gaussian_parameter(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,
                                                input_variables:cdf.input_variables,computed_variables:cdf.computed_variables,
                                                correlation_matrices:cf.correlation_functions)->torch.Tensor:
    """
    Calculates the right-hand side (RHS) of the equation of motion for the non-Gaussian parameter.

    Returns:
    - time_derivative_lambda_bar : 
    TYPE: complex numpy array
    SIZE: N_b x N_f
    Matrix which stores the time derivative of the non-Gaussian parameter.
    """
    N_f = input_variables.N_f
    N_b = input_variables.N_b
    correlation_mat_for_non_gaussian_parameters = correlation_matrices.correlation_mat_for_non_gaussian_parameters
    kappa_1_mat = hdm.rhs_var_par_eqn_of_motion_lambda_real_time(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,
                                                        correlation_matrices)           

    # print(" cond value for correlation matrix is : ", torch.linalg.cond(correlation_mat_for_non_gaussian_parameters))
    cdf.log_and_print(" cond value for correlation matrix is : "+str(torch.linalg.cond(correlation_mat_for_non_gaussian_parameters)) )
    # time_derivative_lambda_bar = torch.linalg.lstsq(correlation_mat_for_non_gaussian_parameters , kappa_1_mat, rcond = 1e-13, driver = 'gelss' )      
    time_derivative_lambda_bar = torch.linalg.lstsq(correlation_mat_for_non_gaussian_parameters.real , kappa_1_mat.real,
                                                    rcond = 1e-13, driver = 'gelss' )

    # We send back the transpose since time_derivative_lambda_bar is a matrix with N_f x N_b dimensions and not N_b x N_f
    # (just check if the above statement is true) 
    # diff = torch.matmul(correlation_mat_for_non_gaussian_parameters.real,time_derivative_lambda_bar[0]) - kappa_1_mat.real
    # print("difference (real):", torch.max(torch.abs(torch.real(diff))) )
    # print("difference (imag) :", torch.max(torch.abs(torch.imag(diff))) )

    final_mat = time_derivative_lambda_bar[0].to(dtype =torch.complex128)
    return(final_mat.T) # type: ignore

def equation_of_motion_for_bosonic_averages(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                            input_variables:cdf.input_variables,computed_variables:cdf.computed_variables,
                                            correlation_matrices:cf.correlation_functions):

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 
    Here we also calculate the contribution of the bosonic average derivatives to the phase term (i.e.propto h_{\\delta} and O_{\\delta}).

    (evol equation result delta_r, phase term)
    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 1 x 2N_b, 1
    """
    N_b = input_variables.N_b

    sigma = torch.from_numpy(np.kron([[0,1],[-1,0]],np.eye(N_b))).to(torch.complex128)
    h_delta_matrix = hdm.h_delta(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,correlation_matrices)
    O_delta_mat = ngtd.O_delta(time_derivative_lambda_bar,Gamma_m,input_variables,correlation_matrices)
    h_delta_t_matrix = h_delta_matrix- 1j*O_delta_mat
    # sigma = np.kron([[0,1],[-1,0]],np.eye(N_b))

    # h_delta_matrix = gf.h_delta(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)
    # O_delta_mat = gf.O_delta(time_derivative_lambda_bar,Gamma_m,input_variables)
    # delta_r_tensor = torch.tensor(delta_r,dtype=torch.complex128)

    # final_mat = torch.matmul(sigma, h_delta_matrix) - complex(0,1)*torch.matmul(sigma,O_delta_mat)
    # final_mat = torch.matmul(sigma, h_delta_matrix- 1j*O_delta_mat)
    final_mat = torch.matmul(sigma, h_delta_t_matrix)

    phase_term = (1/4.0)*torch.matmul(delta_r,h_delta_matrix-complex(0,1)*O_delta_mat)
    return([final_mat,phase_term])

def equation_of_motion_for_bosonic_covariance(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                              input_variables:cdf.input_variables,computed_variables:cdf.computed_variables,
                                              correlation_matrices:cf.correlation_functions):

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 
    Here we also calculate the contribution of the O_b terms to the computation of the phase term.

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 2N_b x 2N_b, phase term contribution
    """

    N_b = input_variables.N_b
    sigma = torch.kron( torch.tensor([[0,1],[-1,0]],dtype=torch.complex128),   torch.eye(N_b,dtype=torch.complex128))

    h_b_matrix = hdm.h_b(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,correlation_matrices)
    O_b_matrix = ngtd.O_b(time_derivative_lambda_bar,input_variables,correlation_matrices)
    h_b_t_matrix = h_b_matrix- 1j*O_b_matrix

    # final_mat = (torch.matmul(sigma,torch.matmul(h_b_matrix-complex(0,1)*O_b_matrix,Gamma_b)) \
    #             - torch.matmul( Gamma_b,torch.matmul(h_b_matrix-complex(0,1)*O_b_matrix,sigma)) )
    final_mat = (torch.matmul(sigma,torch.matmul(h_b_t_matrix,Gamma_b))
                - torch.matmul( Gamma_b,torch.matmul(h_b_t_matrix,sigma)) )
    
    phase_term = 1/4.0*torch.trace(O_b_matrix) -1j/4.0*torch.trace(torch.matmul(torch.matmul(Gamma_b, sigma), O_b_matrix) )
    return([final_mat,phase_term])

def equation_of_motion_for_fermionic_covariance(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                                input_variables:cdf.input_variables,computed_variables:cdf.computed_variables,
                                                correlation_matrices:cf.correlation_functions):

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 
    Here we also calculate the contribution of the O_m terms to the computation of the phase term.

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 2N_b x 2N_b , phase term contribution
    
    """
    h_m_matrix = hdm.h_m(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,correlation_matrices)
    O_m_matrix = ngtd.O_m(time_derivative_lambda_bar,delta_r,input_variables,correlation_matrices)
    h_m_t_matrix = h_m_matrix- 1j*O_m_matrix

    final_mat = torch.matmul(h_m_t_matrix, Gamma_m) - torch.matmul( Gamma_m,h_m_t_matrix) 

    phase_term = 1j/4.0*torch.trace(O_m_matrix) - 1/4.0*torch.trace(torch.matmul(Gamma_m, O_m_matrix) ) 
    return([final_mat,phase_term])

##############################################################################
##############################################################################