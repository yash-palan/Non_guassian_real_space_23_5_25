# -*- coding: utf-8 -*-
"""
Created on  Sep 13, 2024

@author: Yash_palan

File containing the class definition for the input variables and the computed variables class

------------------------------------------
------------------------------------------

Comparison with the 11/7/24 version:
This one assumes that the input variables gamma_k and lmbda_k are normalised with the volume, 
i.e. in terms of the old notation, we have 
lmbda_{in this file} = lmbda/sqrt(volume)
gamma_{in this file} = gamma/sqrt(volume)

due to which we do not need to keep the volume factor everywhere, which hopefully makes the 
RK4 method more stable.
"""
##############################################################################
##############################################################################
import numpy as np
import torch
from Common_codes import generic_codes_20_3_25 as gcs
from Common_codes import class_defn_file_20_3_25 as cdf
from Common_codes import correlation_functions_file_20_3_25 as cf
##############################################################################
##############################################################################
# Energy expectation value computation function

# Global variables
c_dagger_c_mat = None
c_dagger_c_dagger_mat = None
c_c_mat = None
c_c_dagger_mat = None

def initialize_global_matrices(Gamma_m:torch.Tensor, N_f:int):
    global c_dagger_c_mat, c_dagger_c_dagger_mat, c_c_mat, c_c_dagger_mat

    c_dagger_c_mat = cf.c_dagger_c_expectation_value_matrix_creation(Gamma_m, N_f)
    c_dagger_c_dagger_mat = cf.c_dagger_c_dagger_expectation_value_matrix_creation(Gamma_m, N_f)
    c_c_mat = cf.c_c_expectation_value_matrix_creation(Gamma_m, N_f)
    c_c_dagger_mat = cf.c_c_dagger_expectation_value_matrix_creation(Gamma_m, N_f)

def energy_expectation_value(delta_r_tensor:np.ndarray,Gamma_b_tensor:np.ndarray,Gamma_m_tensor:np.ndarray,input_variables:cdf.input_variables,computed_variables:cdf.computed_variables):
    """
    This function computes the energy expectation value for the system

    ------------
    Returns:
    ------------
    Type: complex value (actually should be completely real, but we store in the complex type)
    Size: 1 x 1

    """
    J_i_j_matrix = computed_variables.J_i_j_mat
    Ve_i_J_matrix = computed_variables.Ve_i_j_mat
    delta_gamma_tilde_matrix = computed_variables.delta_gamma_tilde_mat
    omega_bar_matrix = computed_variables.omega_bar_mat
    chemical_potential = torch.diag( computed_variables.chemical_potential(input_variables) )

    if(delta_r_tensor.dtype != torch.complex128):
        raise Exception("The delta_r_tensor is not a complex numpy array. Please check the computation of the energy expectation value.")
    if(Gamma_b_tensor.dtype != torch.complex128):
        raise Exception("Gamma_b is not a complex numpy array. Please check the computation of the energy expectation value.")

    # Correlation matrix creation functions (assumed to be globally stored from now on)
    # c_dagger_c_mat = cf.c_dagger_c_expectation_value_matrix_creation(Gamma_m,input_variables.N_f)
    # c_dagger_c_dagger_mat = cf.c_dagger_c_dagger_expectation_value_matrix_creation(Gamma_m,input_variables.N_f)
    # c_c_mat = cf.c_c_expectation_value_matrix_creation(Gamma_m,input_variables.N_f)

    # Computing of the energy expectation value
    val = (torch.einsum('ij,ij->',J_i_j_matrix,c_dagger_c_mat) - torch.einsum('i,ii->',chemical_potential, c_dagger_c_mat) 
           + 1/4.0* torch.einsum('i,ij,j->',delta_r_tensor,omega_bar_matrix,delta_r_tensor) + 1/4.0*torch.trace(omega_bar_matrix@Gamma_b_tensor)
           + torch.einsum('ki,k,i->',delta_gamma_tilde_matrix,delta_r_tensor,torch.diag(c_dagger_c_mat)) 
           + 0.5*torch.einsum('ij,ii,jj->',Ve_i_J_matrix,c_dagger_c_mat,c_dagger_c_mat)
           - 0.5*torch.einsum('ij,ij,ji->',Ve_i_J_matrix,c_dagger_c_mat,c_dagger_c_mat)
           + 0.5*torch.einsum('ij,ij,ji->',Ve_i_J_matrix,c_dagger_c_dagger_mat,c_c_mat)
           - 1/4.0 *torch.sum(torch.diag(omega_bar_matrix))     # This is the constant energy sgift term. However, it becomes essential to get this correct for the phase shoft and hence check this once again
           )
    if(val.shape == (1,)):
        raise Exception("The energy expectation value is not a single value. Please check the computation of the energy expectation value.")
    
    return(val)


##############################################################################
##############################################################################
# Bosoic average derivative matrix

def h_delta(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,input_variables:cdf.input_variables,computed_variables:cdf.computed_variables)->torch.Tensor:
    """
    Returns:
    Type : complex torch tensor
    Size : (2N_b,)
    """    
    c_dagger_c_expectation_value_mat = c_dagger_c_dagger_mat

    if(c_dagger_c_expectation_value_mat.dtype != torch.complex128):
        raise Exception("The c_dagger_c_expectation_value_mat is not a complex128 torch array. Please check the computation of the energy expectation value.")
    
    if(delta_r.dtype != torch.complex128):
        raise Exception("The delta_r_tensor is not a complex numpy array. Please check the computation of the energy expectation value.")
    if(Gamma_b.dtype != torch.complex128):
        raise Exception("Gamma_b is not a complex numpy array. Please check the computation of the energy expectation value.")
    if(Gamma_m.dtype != torch.complex128):
        raise Exception("Gamma_b is not a complex numpy array. Please check the computation of the energy expectation value.")

    final_mat = torch.zeros(2*input_variables.N_b,dtype=torch.complex128)
    final_mat.add_(-2j*torch.einsum('ij,ijk,ij->k',computed_variables.J_i_j_mat,computed_variables.alpha_bar_mat,c_dagger_c_expectation_value_mat))
    final_mat.add_(2.0*torch.einsum('ki,i->k',computed_variables.delta_gamma_tilde_mat,torch.diag(c_dagger_c_expectation_value_mat) ))
    final_mat.add_(torch.einsum('kl,l->k', computed_variables.omega_bar_mat ,  delta_r ) )

    return(final_mat)
##############################################################################
##############################################################################
# Bosonic covariance matrix derivative functions

def h_b(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,input_variables:cdf.input_variables,computed_variables:cdf.computed_variables)->torch.Tensor:
    # Seems to be working fine. All run time errors have been removed. Yash 11/5/24 00:18 
    """
     Description
    ------------------
    ------------------
    This function creates the h_b matrix. 
    This is the matrix stores the functional derivatives of the Hamiltonian with respect to 
    the Bosonic covariance matrix (\\Gamma_b).
    
    Returns
    -------
    final_mat :     TYPE: complex numpy array
                    SIZE: 2N_b x 2N_b
                    DESCRIPTION: This is the final matrix that is returned.
    """    
    N_f = input_variables.N_f  

    # The breakdown is done just for the sake of memory management
    omega_bar_mat = computed_variables.omega_bar_mat
    c_dagger_c_expectation_value_mat = c_dagger_c_mat
    J_i_j_mat = computed_variables.J_i_j_mat
    alpha_bar_matrix = computed_variables.alpha_bar_mat
    
    if(c_dagger_c_expectation_value_mat.dtype != torch.complex128):
        raise Exception("The c_dagger_c_expectation_value_mat is not a complex128 torch array. Please check the computation of the energy expectation value.")    
    if(delta_r.dtype != torch.complex128):
        raise Exception("The delta_r_tensor is not a complex numpy array. Please check the computation of the energy expectation value.")
    if(Gamma_b.dtype != torch.complex128):
        raise Exception("Gamma_b is not a complex numpy array. Please check the computation of the energy expectation value.")
    if(Gamma_m.dtype != torch.complex128):
        raise Exception("Gamma_b is not a complex numpy array. Please check the computation of the energy expectation value.")

    # temp_mat_1 = J_i_j_mat*c_dagger_c_expectation_value_mat
    # temp_mat_2 = torch.einsum('ijk,ij->ijk', alpha_bar_matrix,temp_mat_1)
    # final_mat = torch.einsum('ijk,ijl->kl', alpha_bar_matrix,temp_mat_2)
    
    # # final_mat = torch.einsum('ijk,ijl,ij->kl', alpha_bar_matrix,alpha_bar_matrix,temp_mat_1)

    # del temp_mat_1
    # del temp_mat_2

    # final_mat.mul_(-2.0)                       # In-place multiplication to save memory
    # # omega_bar_transposed = torch.transpose(omega_bar_mat, 0, 1)
    # # final_mat.add_(0.5*omega_bar_transposed)    # In-place addition to save memory
    # final_mat.add_(omega_bar_mat)           # In-place addition to save memory
    
    final_mat = omega_bar_mat - 2*torch.einsum("ij,ij,ijk,ijl->kl", J_i_j_mat, c_dagger_c_expectation_value_mat, alpha_bar_matrix, alpha_bar_matrix)

    return(final_mat)

##############################################################################
##############################################################################
# Functions which compute the partial derivatives of the Energy functional

def epsilon_i_j(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,input_variables:cdf.input_variables,computed_variables:cdf.computed_variables)->torch.Tensor:

    """
    Description
    This function creates the h_b matrix. 
    This is the matrix stores the functional derivatives of the Hamiltonian with respect to 
    the Bosonic covariance matrix (\\Gamma_b).

    Returns:
    final_array: 
        TYPE: complex numpy array
        SIZE: N_f x N_f
        DESCRIPTION: This is the final matrix that is returned.
    -----------------------------
    -----------------------------
    """
    Ve_real_matrix = computed_variables.Ve_i_j_mat
    c_dagger_c_expectation_value_mat = c_dagger_c_dagger_mat

    # Yash 9/9/24 : Just for better memory management (I believe/hope). This should not cause any change in the actual output.
    final_mat = torch.zeros(input_variables.N_f,input_variables.N_f,dtype=torch.complex128)
    final_mat.add_(computed_variables.J_i_j_mat)
    final_mat.add_(-computed_variables.chemical_potential(input_variables)) 
    final_mat.add_(  torch.diag( 
                                torch.einsum( 'ki,k->i',computed_variables.delta_gamma_tilde_mat,delta_r ) 
                                + torch.einsum('ij,j->i',Ve_real_matrix,torch.diag(c_dagger_c_expectation_value_mat) ) 
                                )  
                )  
    final_mat.add_( -torch.einsum('ij,ji->ij',Ve_real_matrix,c_dagger_c_expectation_value_mat) )
                
    return(final_mat)

def Delta_for_h_m(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,input_variables:cdf.input_variables,computed_variables:cdf.computed_variables)->torch.Tensor:
    # Update : Yash 5/6/24 : Due to the change in the definition of Ve, appropriate changes in the equation for  
    #                          epsilon_i_j have been made. Check with the writeup for the same.
    """    
    -----------------------------
    -----------------------------
    Returns:
    -----------------------------
    -----------------------------
    final_mat: 
        TYPE: complex numpy array
        SIZE: N_f x N_f
        DESCRIPTION: This is the final matrix that is returned.

    
    """
    # Ve_real_matrix = 0.5*(computed_variables.Ve_i_j_mat + computed_variables.Ve_i_j_mat.T)
    Ve_real_matrix = computed_variables.Ve_i_j_mat
    # c_c_expectation_value_mat = cf.c_c_expectation_value_matrix_creation(Gamma_m,input_variables.N_f)
    c_c_expectation_value_mat = c_c_mat
    
    # # Remember that since Delta = V^{e}_{i,j} *<c_{j}c_{i}> = -V^{e}_{i,j} *<c_{i}c_{j}>  

    # Remark : Just check the sign here once again. Not sure of the sign.
    final_mat = torch.einsum('ij,ji->ij',Ve_real_matrix,c_c_expectation_value_mat)  
    return(final_mat)    

##############################################################################
##############################################################################
# Fermionic covariance derivative matrix
def h_m(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,input_variables:cdf.input_variables,computed_variables:cdf.computed_variables)->torch.Tensor:
    """
    ------------------
    ------------------
    Description
    ------------------
    ------------------
    This function creates the h_f matrix. 
    This is the matrix stores the functional derivatives of the Hamiltonian with respect to 
    the Fermionic covariance matrix (\\Gamma_f).

    Returns
    -------
    final_mat :     TYPE: numpy array
                    SIZE: 2N_f x 2N_f
                    DESCRIPTION: This is the final matrix that is returned.

    """   
    # Defining the epsilon and delta matrices for the computation of the h_m matrix
    epsilon_i_j_mat = epsilon_i_j(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)
    Delta_mat_for_h_m = Delta_for_h_m(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)

    # All additions are done in place so as to preserve memory 
    # Adding the epsilon part of the h_m matrix
    final_mat = 0.5*torch.kron(torch.tensor([[-1j,1],[-1,-1j]]),epsilon_i_j_mat)

    # Adding the epsilon transpose part of the h_m matrix
    final_mat.add_( 0.5*torch.kron(torch.tensor([[1j,1],[-1,1j]]),epsilon_i_j_mat.t().contiguous() ) )  

    # Adding the Delta part of the h_m matrix
    final_mat.add_( 0.5*torch.kron(torch.tensor((  [  [-1j,-1],  [-1,1j]  ]  )),Delta_mat_for_h_m ) )  
    
    # Adding the Delta dagger part of the h_m matrix
    final_mat.add_( 0.5*torch.kron(torch.tensor([  [ -1j, 1  ],[  1 ,  1j ]  ]),Delta_mat_for_h_m.t().conj().contiguous()  )  )  

    return(final_mat)

##############################################################################
##############################################################################
# Functions needed for the computation of the Equation of motion of the Variational parameters (Imaginary time evolution)

def rhs_var_par_eqn_of_motion(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,input_variables:cdf.input_variables,
                              computed_variables:cdf.computed_variables,correlation_matrix=None)->np.ndarray:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    Computes the right-hand side of the variational partial differential equation of motion for kappa_1 
    -----------------------------
    -----------------------------
    Returns:
    -----------------------------
    -----------------------------
    ndarray: The computed matrix.
    TYPE: complex numpy array
    SIZE: N_f x N_b 
    DESCRIPTION:

    """
    # Defining number of the fermionic and bosonic particles     
    N_b = input_variables.N_b
    N_f = input_variables.N_f


    # Sending all of the tensors to the GPU is it exists, else to the CPU
    sigma_mat = torch.from_numpy(  sigma(N_b,dtype="complex")  ).to(device=device)
    delta_gamma_tilda_mat = computed_variables.delta_gamma_tilde_mat.to(device=device)
    alpha_bar_mat = computed_variables.alpha_bar_mat.to(device=device)
    J_i_j_mat = computed_variables.J_i_j_mat.to(device=device)
    denisty_density_anticommutator_connected_correlation_mat = cf.density_density_anticommutator_connected_correlation_matrix_creation(Gamma_m,N_f).to(device=device)
    c_dagger_c_expectation_value_mat = cf.c_dagger_c_expectation_value_matrix_creation(Gamma_m,N_f).to(device=device)
    Gamma_b_tensor = torch.from_numpy(Gamma_b).to(device=device)

    if(correlation_matrix is None):
        correlation_mat = cf.correlation_matrix_creation(Gamma_m,N_f)
        if(correlation_mat.dtype != torch.complex128):
            correlation_mat = correlation_mat.to(device=device,dtype=torch.complex128)
            
    if(type(correlation_matrix) is np.ndarray):
        correlation_mat = torch.from_numpy(correlation_matrix).to(device=device,dtype=torch.complex128)
    # else:
    #     raise Exception("Wrong type of the correlation matrix. Please check the type of the correlation matrix.")

#   Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
    # Computing the values of the return matrix
    rhs_mat_3 = torch.einsum('kl,li,ij->kj',Gamma_b_tensor,delta_gamma_tilda_mat,correlation_mat )
    # print("Computed rhs_mat_3 successfully")

#   Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
    rhs_mat_2 = torch.einsum('kl,jil,ji->kj',sigma_mat,alpha_bar_mat,torch.real(J_i_j_mat*c_dagger_c_expectation_value_mat).to(dtype=torch.complex128) )
    # print("Computed rhs_mat_2 successfully")

#   Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
    rhs_mat_1 = torch.einsum('kl,iml,im,jim->kj',Gamma_b_tensor,alpha_bar_mat,J_i_j_mat,denisty_density_anticommutator_connected_correlation_mat)
    # print("Computed rhs_mat_1 successfully")

    # Combining all of the above matrices into one final matrix which is then transposed and sent back to the cpu to be converted into
    # a numpy array
    final_mat = torch.transpose((-0.5*1j*rhs_mat_1 + rhs_mat_2 + rhs_mat_3),0,1).cpu().numpy()

    # Sending all of the arrays back to the CPU (in case they are on the GPU) 
    # This is being done since I am not sure of the way the computations in one run may affect another run due to the fact that
    # the GPU memory is not cleared after each run, and also due to how arguments of a function might be mutable, leading to more
    # confusion of accessing (especially when being sent to the GPU)

    # However, I just need to be careful of the computation time overhead that this passing up and down might lead to 
    # since it can lead to a large overhead in the computation time.
    if(device.type=="cuda"):
        # sigma_mat = torch.from_numpy(  sigma(N_b,dtype="complex")  ).to(device=device)
        delta_gamma_tilda_mat = computed_variables.delta_gamma_tilde_mat.cpu()
        alpha_bar_mat = computed_variables.alpha_bar_mat.cpu()
        J_i_j_mat = computed_variables.J_i_j_mat.cpu()
        denisty_density_anticommutator_connected_correlation_mat = denisty_density_anticommutator_connected_correlation_mat.cpu()
        c_dagger_c_expectation_value_mat = c_dagger_c_expectation_value_mat.cpu()
        correlation_mat = correlation_mat.cpu()
        Gamma_b_tensor = Gamma_b_tensor.cpu()
        torch.cuda.empty_cache()
        
    if( True in set(np.isinf(np.reshape(final_mat,(-1)))) or True in set(np.isnan(np.reshape(final_mat,(-1))))  ):
        raise Exception("We get a NaN or an Inf in the final matrix. Please check the computation of the matrix.")

    return(final_mat[:,0:N_b])

##############################################################################
##############################################################################
# Functions needed for the computation of the Equation of motion of the Variational parameters (Real time evolution)


##############################################################################
##############################################################################
# Functions needed for the computation of the Equation of motion of the Variational parameters (Imaginary time evolution)

# def rhs_var_par_eqn_of_motion(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,input_variables:cdf.input_variables,
#                               computed_variables:cdf.computed_variables,correlation_matrix=None)->np.ndarray:
#     """
#     -----------------------------
#     -----------------------------
#     DESCRIPTION
#     -----------------------------
#     -----------------------------
#     Computes the right-hand side of the variational partial differential equation of motion for kappa_1 (and 
#     kappa_2 as well).
#     -----------------------------
#     -----------------------------
#     Returns:
#     -----------------------------
#     -----------------------------
#     ndarray: The computed matrix.
#     TYPE: complex numpy array
#     SIZE: N_f x N_b 
#     DESCRIPTION:

#     """
#     # Defining number of the fermionic and bosonic particles     
#     N_b = input_variables.N_b
#     N_f = input_variables.N_f

#     # Selecting the device (preferably GPU if available)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Sending all of the tensors to the GPU is it exists, else to the CPU
#     sigma_mat = torch.from_numpy(  sigma(N_b,dtype="complex")  ).to(device=device)
#     delta_gamma_tilda_mat = computed_variables.delta_gamma_tilde_mat.to(device=device)
#     alpha_bar_mat = computed_variables.alpha_bar_mat.to(device=device)
#     J_i_j_mat = computed_variables.J_i_j_mat.to(device=device)
#     denisty_density_anticommutator_connected_correlation_mat = cf.density_density_anticommutator_connected_correlation_matrix_creation(Gamma_m,N_f).to(device=device)
#     c_dagger_c_expectation_value_mat = cf.c_dagger_c_expectation_value_matrix_creation(Gamma_m,N_f).to(device=device)
#     Gamma_b_tensor = torch.from_numpy(Gamma_b).to(device=device)

#     if(correlation_matrix is None):
#         correlation_mat = cf.correlation_matrix_creation(Gamma_m,N_f)
#         if(correlation_mat.dtype != torch.complex128):
#             correlation_mat = correlation_mat.to(device=device,dtype=torch.complex128)
            
#     if(type(correlation_matrix) is np.ndarray):
#         correlation_mat = torch.from_numpy(correlation_matrix).to(device=device,dtype=torch.complex128)
#     # else:
#     #     raise Exception("Wrong type of the correlation matrix. Please check the type of the correlation matrix.")

# #   Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
#     # Computing the values of the return matrix
#     rhs_mat_3 = torch.einsum('kl,li,ij->kj',Gamma_b_tensor,delta_gamma_tilda_mat,correlation_mat )
#     # print("Computed rhs_mat_3 successfully")

# #   Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
#     rhs_mat_2 = torch.einsum('kl,jil,ji->kj',sigma_mat,alpha_bar_mat,torch.real(J_i_j_mat*c_dagger_c_expectation_value_mat).to(dtype=torch.complex128) )
#     # print("Computed rhs_mat_2 successfully")

# #   Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
#     rhs_mat_1 = torch.einsum('kl,iml,im,jim->kj',Gamma_b_tensor,alpha_bar_mat,J_i_j_mat,denisty_density_anticommutator_connected_correlation_mat)
#     # print("Computed rhs_mat_1 successfully")

#     # Combining all of the above matrices into one final matrix which is then transposed and sent back to the cpu to be converted into
#     # a numpy array
#     final_mat = torch.transpose((-0.5*1j*rhs_mat_1 + rhs_mat_2 + rhs_mat_3),0,1).cpu().numpy()

#     # Sending all of the arrays back to the CPU (in case they are on the GPU) 
#     # This is being done since I am not sure of the way the computations in one run may affect another run due to the fact that
#     # the GPU memory is not cleared after each run, and also due to how arguments of a function might be mutable, leading to more
#     # confusion of accessing (especially when being sent to the GPU)

#     # However, I just need to be careful of the computation time overhead that this passing up and down might lead to 
#     # since it can lead to a large overhead in the computation time.
#     if(device.type=="cuda"):
#         # sigma_mat = torch.from_numpy(  sigma(N_b,dtype="complex")  ).to(device=device)
#         delta_gamma_tilda_mat = computed_variables.delta_gamma_tilde_mat.cpu()
#         alpha_bar_mat = computed_variables.alpha_bar_mat.cpu()
#         J_i_j_mat = computed_variables.J_i_j_mat.cpu()
#         denisty_density_anticommutator_connected_correlation_mat = denisty_density_anticommutator_connected_correlation_mat.cpu()
#         c_dagger_c_expectation_value_mat = c_dagger_c_expectation_value_mat.cpu()
#         correlation_mat = correlation_mat.cpu()
#         Gamma_b_tensor = Gamma_b_tensor.cpu()
#         torch.cuda.empty_cache()
        
#     if( True in set(np.isinf(np.reshape(final_mat,(-1)))) or True in set(np.isnan(np.reshape(final_mat,(-1))))  ):
#         raise Exception("We get a NaN or an Inf in the final matrix. Please check the computation of the matrix.")

#     return(final_mat[:,0:N_b])
