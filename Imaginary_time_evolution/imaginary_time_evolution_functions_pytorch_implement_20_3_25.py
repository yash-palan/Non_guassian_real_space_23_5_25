# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:43:50 2023

@author: Yash_palan

This file is used for definining all the functions that we will need for Imaginary time evolution
of the state.

"""
##############################################################################
##############################################################################
# from Common_codes import global_functions_v3_pytorch_implement_24_7_24 as gf
import numpy as np
# from scipy import linalg as la
import torch
from Common_codes import hamiltonian_derivative_matrices_20_3_25 as hdm
from Common_codes import non_gaussian_transform_derivative_20_3_25 as ngtd
from Common_codes import class_defn_file_20_3_25 as cdf
from Common_codes import correlation_functions_file_20_3_25 as cf
##############################################################################
##############################################################################
# First equation of motion functions

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
    kappa_1_mat = hdm.rhs_var_par_eqn_of_motion(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,
                                                        correlation_matrices)           

    print(" cond value for correlation matrix is : ", torch.linalg.cond(correlation_mat_for_non_gaussian_parameters))
    # time_derivative_lambda_bar = torch.linalg.lstsq(correlation_mat_for_non_gaussian_parameters , kappa_1_mat, rcond = 1e-13, driver = 'gelss' )      
    time_derivative_lambda_bar = torch.linalg.lstsq(correlation_mat_for_non_gaussian_parameters.real , kappa_1_mat.real,
                                                rcond = 1e-13, driver = 'gelss' )
    # We send back the transpose since time_derivative_lambda_bar is a matrix with N_f x N_b dimensions and not N_b x N_f
    # (just check if the above statement is true) 
    final_mat = time_derivative_lambda_bar[0].to(dtype =torch.complex128)

    # return(time_derivative_lambda_bar[0].T) # type: ignore
    return(final_mat.T) # type: ignore
    # return(time_derivative_lambda_bar.T)

def equation_of_motion_for_Non_Gaussian_parameter_spin_modes_summed(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,
                                                input_variables:cdf.input_variables,computed_variables:cdf.computed_variables, 
                                                correlation_matrices:cf.correlation_functions,spin_index=True)->torch.Tensor:
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
    # start_time = time.time()
    # print("\n correlation_mat_for_non_gaussian_parameters = ")
    correlation_mat_for_non_gaussian_parameters = correlation_matrices.correlation_mat_for_non_gaussian_parameters
    kappa_1_mat = hdm.rhs_var_par_eqn_of_motion(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,
                                                        correlation_matrices)
    if(spin_index==True):
        spin_summed_correlation_mat = (correlation_mat_for_non_gaussian_parameters[0:N_b,0:N_b] + correlation_mat_for_non_gaussian_parameters[0:N_b,N_b:] 
                                        +correlation_mat_for_non_gaussian_parameters[N_b:,0:N_b]  + correlation_mat_for_non_gaussian_parameters[N_b:,N_b:]
                                        )
        spin_summed_kappa_1_mat = kappa_1_mat[0:N_b,:] + kappa_1_mat[N_b:,:]             
    cond_val = torch.linalg.cond(correlation_mat_for_non_gaussian_parameters)
        
    time_derivative_lambda_bar = torch.linalg.lstsq(spin_summed_correlation_mat,spin_summed_kappa_1_mat, rcond = 1e-13, driver = 'gelss' )      
    print(" ---> condition number : ", cond_val)
    print(" ---> rank = ", time_derivative_lambda_bar[2])

    # We send back the transpose since time_derivative_lambda_bar is a matrix with N_f/2 x N_b dimensions and not N_b x N_f/2
    # (just check if the above statement is true) 

    return(time_derivative_lambda_bar[0].T) # type: ignore
    # return(time_derivative_lambda_bar.T)

def equation_of_motion_for_bosonic_averages(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                            input_variables:cdf.input_variables,computed_variables:cdf.computed_variables,
                                            correlation_matrices:cf.correlation_functions)->torch.Tensor:

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 1 x 2N_b
                This function returns the term given by eqn 19 in the writeup 
                "Ch-summary_of_equations" on GITHUB as the matrix
    """
    N_b = input_variables.N_b

    sigma = torch.from_numpy(np.kron([[0,1],[-1,0]],np.eye(N_b))).to(torch.complex128)
    h_delta_matrix = hdm.h_delta(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,correlation_matrices)
    O_delta_mat = ngtd.O_delta(time_derivative_lambda_bar,Gamma_m,input_variables,correlation_matrices)

    # final_mat = -np.matmul(Gamma_b, h_delta_matrix) - complex(0,1)*np.matmul(sigma,O_delta_mat)
    final_mat = -torch.matmul(Gamma_b, h_delta_matrix) - complex(0,1)*torch.matmul(sigma,O_delta_mat)
    # final_mat = -h_delta_matrix

    return(final_mat)

def equation_of_motion_for_bosonic_covariance(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                              input_variables:cdf.input_variables,computed_variables:cdf.computed_variables,
                                              correlation_matrices:cf.correlation_functions)->torch.Tensor:
    # Working properly. All run time errors have been removed. Yash 11/5/24 14:02
    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 2N_b x 2N_b
                This function returns the term given by eqn 19 in the writeup 
                "Ch-summary_of_equations" on GITHUB as the matrix
    """
    N_b = input_variables.N_b

    #     final_mat = np.zeros((2*N_b,),"complex")    
    sigma = torch.kron( torch.tensor([[0,1],[-1,0]],dtype=torch.complex128),   torch.eye(N_b,dtype=torch.complex128))
    # Gamma_b_tensor = Gamma_b

    h_b_matrix = hdm.h_b(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,correlation_matrices)
    O_b_matrix = ngtd.O_b(time_derivative_lambda_bar,input_variables,correlation_matrices)

    final_mat = (torch.matmul(sigma.t(),torch.matmul(h_b_matrix,sigma)) \
                            -torch.matmul(Gamma_b,torch.matmul(h_b_matrix,Gamma_b)) \
                            -complex(0,1)*torch.matmul(sigma,torch.matmul(O_b_matrix,Gamma_b)) \
                            +complex(0,1)*torch.matmul(Gamma_b,torch.matmul(O_b_matrix,sigma)))

    return final_mat

def equation_of_motion_for_fermionic_covariance(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                                input_variables:cdf.input_variables,computed_variables:cdf.computed_variables,
                                                correlation_matrices:cf.correlation_functions)->torch.Tensor:

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 

    ------------------------------------
    ------------------------------------
    Returns
    ------------------------------------
    ------------------------------------
    final_mat : TYPE: complex numpy array
                SIZE: 2N_b x 2N_b
                This function returns the term given by eqn 19 in the writeup 
                "Ch-summary_of_equations" on GITHUB as the matrix
    """
    h_m_matrix = hdm.h_m(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,correlation_matrices)
    O_m_matrix = ngtd.O_m(time_derivative_lambda_bar,delta_r,input_variables,correlation_matrices)

    final_mat = ( -h_m_matrix - torch.matmul(Gamma_m,torch.matmul(h_m_matrix,Gamma_m)) \
                + complex(0,1)*(torch.matmul(Gamma_m,O_m_matrix)-torch.matmul(O_m_matrix,Gamma_m)) \
        )
    # final_mat = ( -h_m_matrix - torch.matmul(Gamma_m,torch.matmul(h_m_matrix,Gamma_m)) \
    #     )
    
    return(final_mat)
