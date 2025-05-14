# -*- coding: utf-8 -*-
"""
Created on  Sep 13, 2024

@author: Yash_palan

File containing the function which compute correlations of fermions

------------------------------------------
------------------------------------------

Comparison with the 11/7/24 version:
This one assumes that the input variables gamma_k and lmbda_k are normalised with the volume, 
i.e. in terms of the old notation, we have 
lmbda_{in this file} = lmbda/sqrt(volume)
gamma_{in this file} = gamma/sqrt(volume)

due to which we do not need to keep the volume factor everywhere, which hopefully makes the 
RK4 method more stable.

------------------------------------------
------------------------------------------
Yash Remarks: 13/9/24 :  Works properly. Used matlab code to compare the two.

"""
##############################################################################
##############################################################################
import numpy as np
import torch
##############################################################################
##############################################################################
# Basic correlators 

def c_dagger_c_expectation_value_matrix_creation(Gamma_m:torch.Tensor,N_f:int)->torch.Tensor:
        """
        -----------------------------
        -----------------------------
        DESCRIPTION
        -----------------------------
        -----------------------------
        This function computes the correlation function <c^{\\dagger}_{i}c_{j}>_{GS}.
        This is needed for the computation of the equation of motion of the variational parameters.
        -----------------------------
        -----------------------------    
        RETURNS
        -----------------------------
        -----------------------------
        Type : complex pytorch tensor
        Size : N_f x N_f
    
        """
        # Remark: Yash : 6/9/24 : This expression is correct
        return(0.25*(2*torch.eye(N_f,dtype=torch.complex128) - 1j*(Gamma_m[0:N_f,0:N_f] + Gamma_m[N_f:,N_f:] ) + Gamma_m[0:N_f,N_f:] - Gamma_m[N_f:,0:N_f] ) ) 
        
def c_c_dagger_expectation_value_matrix_creation(Gamma_m:torch.Tensor,N_f)->torch.Tensor:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    This function computes the correlation function <c^{\\dagger}_{i}c_{j}>_{GS}.
    This is needed for the computation of the equation of motion of the variational parameters.
    -----------------------------
    -----------------------------    
    RETURNS
    -----------------------------
    -----------------------------
    Type : complex pytorch tensor
    Size : N_f x N_f
    """
    # Remark: Yash : 6/9/24 : This expression is correct
    return(0.25*(2*torch.eye(N_f,dtype=torch.complex128) - 1j*(Gamma_m[0:N_f,0:N_f] + Gamma_m[N_f:,N_f:] ) - Gamma_m[0:N_f,N_f:] + Gamma_m[N_f:,0:N_f] ) )

def c_c_expectation_value_matrix_creation(Gamma_m:torch.Tensor,N_f:int)->torch.Tensor:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    This function computes the correlation function <c^_{i}c_{j}>_{GS}.
    This is needed for the computation of the equation of motion of the variational parameters.
    -----------------------------    
    RETURNS
    -----------------------------
    -----------------------------
    Type : complex pytorch tensor
    Size : N_f x N_f
    """
    # Remark: Yash : 6/9/24 : This expression is correct
    return (  0.25*(-1j*(Gamma_m[0:N_f,0:N_f] - Gamma_m[N_f:,N_f:] ) + (Gamma_m[0:N_f,N_f:] + Gamma_m[N_f:,0:N_f]) ) )

def c_dagger_c_dagger_expectation_value_matrix_creation(Gamma_m:torch.Tensor,N_f:int)->torch.Tensor:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    This function computes the correlation function <c^{\\dagger}_{i}c_{j}>_{GS}.
    This is needed for the computation of the equation of motion of the variational parameters.
    -----------------------------
    -----------------------------    
    RETURNS
    -----------------------------
    -----------------------------
    Type : complex pytorch tensor
    Size : N_f x N_f
    """
    # Remark: Yash : 6/9/24 : This expression is correct
    return( 0.25*(-1j*(Gamma_m[0:N_f,0:N_f] - Gamma_m[N_f:,N_f:] ) - (Gamma_m[0:N_f,N_f:] + Gamma_m[N_f:,0:N_f]) ) )

##############################################################################
##############################################################################
# Functions needed to compute the connected correlation functions

def density_density_anticommutator_connected_correlation_matrix_creation(Gamma_m:torch.Tensor,N_f:int)->torch.Tensor:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    This function computes the correlation function <{ n_j, c{\\dagger}_{i'} c_{j'} } >_{c}.
    This is needed for the computation of the equation of motion of the variational parameters.   
    -----------------------------
    -----------------------------    
    RETURNS
    -----------------------------
    -----------------------------
    final_mat : TYPE: complex pytorch tensor
                SIZE: N_f x N_f x N_f
                DESCRIPTION: This matrix stores the information about the <{ n_j, c{\\dagger}_{i'} c_{j'} } >_{c}.
    """ 
    c_dagger_c_expectation_value_mat = c_dagger_c_expectation_value_matrix_creation(Gamma_m,N_f)
    c_c_expectation_value_mat = c_c_expectation_value_matrix_creation(Gamma_m,N_f)
    c_dagger_c_dagger_expectation_value_mat = c_dagger_c_dagger_expectation_value_matrix_creation(Gamma_m,N_f)
    c_c_dagger_expectation_value_mat = c_c_dagger_expectation_value_matrix_creation(Gamma_m,N_f)

    # Yash 9/9/24: This is to save memory.
    final_mat = torch.zeros(N_f,N_f,N_f,dtype=torch.complex128)
    final_mat.add_(-torch.einsum('ji,jk->jik',c_dagger_c_dagger_expectation_value_mat,c_c_expectation_value_mat))
    final_mat.add_(-torch.einsum('ij,kj->jik',c_dagger_c_dagger_expectation_value_mat,c_c_expectation_value_mat))
    final_mat.add_(torch.einsum('jk,ji->jik',c_dagger_c_expectation_value_mat,c_c_dagger_expectation_value_mat))
    final_mat.add_(torch.einsum('ij,kj->jik',c_dagger_c_expectation_value_mat,c_c_dagger_expectation_value_mat))

    return(final_mat)

def correlation_matrix_creation(Gamma_m:torch.Tensor,N_f:int)->torch.Tensor:
    # Seems to be working fine. All run time errors have been removed. Yash 11/5/24 00:39
    # May need some more checking in terms of the computation of the values
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    Function which computes the correlation matrix <n_{i}n_{j}>_{c}.

    -----------------------------
    -----------------------------
    Parameters:
    -----------------------------
    -----------------------------
    N_f :   Type: int
            Description: Stores the number of fermions in the system.

    Gamma_m :   Type: numpy array
                Size: 2N_f x 2N_f
                Description: This is the matrix storing the values for the Majorana covariances.
    -----------------------------
    -----------------------------
    Returns:
    -----------------------------
    -----------------------------
    A matrix which stores the vlaues of the expectation values < n_i n_j >_c (c = connected diagrams)

    TYPE: complex numpy array
    SIZE: N_f x N_f
    DESCRIPTION: 

    """

    c_dagger_c_expectation_value_mat = c_dagger_c_expectation_value_matrix_creation(Gamma_m,N_f)
    c_c_expectation_value_mat = c_c_expectation_value_matrix_creation(Gamma_m,N_f)
    c_dagger_c_dagger_expectation_value_mat = c_dagger_c_dagger_expectation_value_matrix_creation(Gamma_m,N_f)
    c_c_dagger_expectation_value_mat=c_c_dagger_expectation_value_matrix_creation(Gamma_m,N_f)

    # Yash 9/9/24 : Just for better memory management (I believe/hope). This should not cause any change in the actual output.    
    final_mat = torch.zeros(N_f,N_f,dtype=torch.complex128)
    final_mat.add_(-torch.einsum('ji,ji->ji',c_dagger_c_dagger_expectation_value_mat,c_c_expectation_value_mat))
    final_mat.add_(torch.einsum('ji,ji->ji',c_dagger_c_expectation_value_mat,c_c_dagger_expectation_value_mat) )

    return(final_mat) 

