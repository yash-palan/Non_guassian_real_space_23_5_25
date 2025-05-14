# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:41:53 2023

@author: Yash_palan
"""
############################################
############################################
import numpy as np
import torch
############################################
############################################
def sigma(N_b):
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
    mat_1 = torch.tensor([[0.0,1.0],[-1.0,0.0]],dtype=torch.complex128)
    mat_2 = torch.eye(N_b,dtype=torch.complex128)
    final_mat=torch.kron(mat_1, mat_2)
    return(final_mat)

def check_majorana_covariance_matrix(Gamma_m: torch.Tensor):
    # Gamma_m^{2} = -I
    max_deviation_pure_state = torch.max(torch.abs(torch.matmul(Gamma_m, Gamma_m) + torch.eye(len(Gamma_m), dtype=Gamma_m.dtype)) )
    if max_deviation_pure_state > 1e-4:
        max_index = torch.argmax(torch.abs(torch.matmul(Gamma_m, Gamma_m) + torch.eye(len(Gamma_m), dtype=Gamma_m.dtype)))
        print(f"WARNING: Max Gamma_m^2 + I: {max_deviation_pure_state:.1e} at index {max_index}")
        # raise Exception("The condition for a Pure Fermionic state (Gamma_m^{2} = -I) is not satisfied.")
    # Gamma_m = -Gamma_m^{T}
    max_deviation_anti_symmetry = torch.max(torch.abs(Gamma_m + Gamma_m.T))
    if max_deviation_anti_symmetry > 1e-10:
        max_index_anti_symmetry = torch.argmax(torch.abs(Gamma_m + Gamma_m.T))
        print(f"WARNING: Max Gamma_m + Gamma_m^T: {max_deviation_anti_symmetry:.1e} at index {max_index_anti_symmetry}")
        # raise Exception("The condition for the Majorana covariance matrix being Anti-Symmetric is not satisfied.")
    # Gamma_m = real
    max_deviation_real = torch.max(torch.abs(torch.imag(Gamma_m)))
    if max_deviation_real > 1e-10:
        max_index_real = torch.argmax(torch.abs(torch.imag(Gamma_m)))
        print(f"WARNING: Largest absolute value in the imaginary part of Gamma_m: {max_deviation_real:.1e} at index {max_index_real}")
        # raise Exception("The condition for the Majorana covariance matrix being Real is not satisfied.")
    return

def check_bosonic_quadrature_covariance_matrix(Gamma_b: torch.Tensor):

    N_b = int(Gamma_b.shape[0]/ 2)
    sigma_mat = sigma(N_b)
    # Gamma_b sigma_mat Gamma_b^{T} = sigma_mat
    deviation = torch.abs(torch.matmul(Gamma_b, torch.matmul(sigma_mat, Gamma_b.T)) - sigma_mat)

    max_deviation = torch.max(deviation)
    if max_deviation > 1e-6:
        max_index = torch.argmax(deviation)
        print(f"WARNING: symplectic condition: {max_deviation:.1e} at index {max_index}")
        # raise Exception("The condition for the quadrature covariance matrix to be Symplectic is not satisfied.")
    # Gamma_b = Gamma_b^{T}

    max_deviation_symmetric = torch.max(torch.abs(Gamma_b.T - Gamma_b))
    if max_deviation_symmetric > 1e-10:
        max_index_symmetric = torch.argmax(torch.abs(Gamma_b.T - Gamma_b))
        print(f"Largest deviation from symmetric condition: {max_deviation_symmetric:.1e} at index {max_index_symmetric}")
        # raise Exception("The condition for the bosonic quadrature covariance matrix being Symmetric is not satisfied.")
    # Gamma_b = real
    
    max_deviation_real = torch.max(torch.abs(torch.imag(Gamma_b)))
    if max_deviation_real > 1e-10:
        max_index_real = torch.argmax(torch.abs(torch.imag(Gamma_b)))
        print(f"Largest absolute value in the imaginary part of Gamma_b: {max_deviation_real:1e} at index {max_index_real}")
        # raise Exception("The condition for the bosonic quadrature covariance matrix being Real is not satisfied.")
    
    return

def check_bosonic_quadrature_average_matrix(Delta_r: torch.Tensor):
    # Delta_r = real
    max_imag_delta = torch.max(torch.abs(torch.imag(Delta_r)))
    if max_imag_delta > 1e-10:
        max_index_imag_delta = torch.argmax(torch.abs(torch.imag(Delta_r)))
        print(f"WARNING: Largest absolute value in the imaginary part of Delta_r: {max_imag_delta:.1e} at index {max_index_imag_delta}")
        # raise Exception("The condition for the bosonic_quadrature_average covariance matrix being Real is not satisfied.")
    return