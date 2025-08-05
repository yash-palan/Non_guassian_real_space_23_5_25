# -*- coding: utf-8 -*-
"""
Created on  July 27, 2025

@author: Yash_palan

File containing the class definition for the input variables and the computed variables class

------------------------------------------
------------------------------------------
"""
##############################################################################
##############################################################################
from typing import Union
import numpy as np
import torch
from Common_codes import damped_hamiltonian_codes as dhc
##############################################################################
##############################################################################

def damped_term_equation_of_motion_for_bosonic_averages(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                                    damped_hamiltonian):

    """
    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 1 x 2N_b, 1
    """
    N_b = damped_hamiltonian.N_b 

    sigma = torch.from_numpy(np.kron([[0,1],[-1,0]],np.eye(N_b))).to(torch.complex128)
    
    h_delta_mat = damped_hamiltonian.h_delta_damped_mat

    final_mat = torch.matmul(Gamma_b, h_delta_mat)

    # phase_term = (1/4.0)*torch.matmul(delta_r,h_delta_matrix-complex(0,1)*O_delta_mat)
    return(final_mat)

def damped_term_equation_of_motion_for_bosonic_covariances(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                                    damped_hamiltonian):

    """
    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 1 x 2N_b, 1
    """
    N_b = damped_hamiltonian.N_b 

    sigma = torch.from_numpy(np.kron([[0,1],[-1,0]],np.eye(N_b))).to(torch.complex128)
    
    h_b_mat = damped_hamiltonian.h_b_damped_mat

    # final_mat = Gamma_b @ h_b_mat @ Gamma_b - sigma.T.contiguous() @ h_b_mat @ sigma
    final_mat = Gamma_b @ h_b_mat @ Gamma_b + sigma @ h_b_mat @ sigma

    # phase_term = (1/4.0)*torch.matmul(delta_r,h_delta_matrix-complex(0,1)*O_delta_mat)
    return(final_mat)

def damped_term_equation_of_motion_for_fermionic_covariance(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,time_derivative_lambda_bar:torch.Tensor,
                                                    damped_hamiltonian):

    """
    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 1 x 2N_b, 1
    """    
    h_m_mat = damped_hamiltonian.h_m_damped_mat

    final_mat = torch.matmul(torch.matmul(Gamma_m, h_m_mat),Gamma_m) + h_m_mat

    # phase_term = (1/4.0)*torch.matmul(delta_r,h_delta_matrix-complex(0,1)*O_delta_mat)
    return(final_mat)

