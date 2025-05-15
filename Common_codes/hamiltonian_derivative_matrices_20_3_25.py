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


def energy_expectation_value(delta_r_tensor:torch.Tensor,Gamma_b_tensor:torch.Tensor,Gamma_m_tensor:torch.Tensor,
                            input_variables:cdf.input_variables,
                            computed_variables:cdf.computed_variables,
                            correlation_matrices:cf.correlation_functions)->torch.Tensor:
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
    # c_dagger_c_mat = correlation_matrices.c_c_dagger_mat
    c_dagger_c_mat = correlation_matrices.c_dagger_c_mat
    c_dagger_c_dagger_mat = correlation_matrices.c_dagger_c_dagger_mat
    c_c_mat = correlation_matrices.c_c_mat

    phonon_energy = 1/4.0* torch.einsum('i,ij,j->',delta_r_tensor,omega_bar_matrix,delta_r_tensor) + 1/4.0*torch.trace(omega_bar_matrix@Gamma_b_tensor)- 1/4.0 *torch.sum(torch.diag(omega_bar_matrix)) 

    electron_kinetic_energy = (torch.einsum('ij,ij->',J_i_j_matrix,c_dagger_c_mat) - torch.einsum('i,ii->',chemical_potential, c_dagger_c_mat) )
    
    electron_electron_interaction_energy = (0.5*torch.einsum('ij,ii,jj->',Ve_i_J_matrix,c_dagger_c_mat,c_dagger_c_mat)
                                            - 0.5*torch.einsum('ij,ij,ji->',Ve_i_J_matrix,c_dagger_c_mat,c_dagger_c_mat)
                                            + 0.5*torch.einsum('ij,ij,ji->',Ve_i_J_matrix,c_dagger_c_dagger_mat,c_c_mat))
    
    electron_phonon_interaction_energy = torch.einsum('ki,k,i->',delta_gamma_tilde_matrix,delta_r_tensor,torch.diag(c_dagger_c_mat)) 
    print(" Phonon energy: ",phonon_energy)
    print(" Electron kinetic energy: ",electron_kinetic_energy)
    print(" Electron electron interaction energy: ",electron_electron_interaction_energy)
    print(" Electron phonon interaction energy: ",electron_phonon_interaction_energy)

    with open('data/phonon_energy.dat','a') as file:
        file.write(str(phonon_energy.item().real)+'\n')

    with open('data/electron_kinetic_energy.dat','a') as file:
        file.write(str(electron_kinetic_energy.item().real)+'\n')
    
    with open('data/electron_electron_interaction_energy.dat','a') as file:
        file.write(str(electron_electron_interaction_energy.item().real)+'\n')

    with open('data/electron_phonon_interaction_energy.dat','a') as file:
        file.write(str(electron_phonon_interaction_energy.item().real)+'\n')

    # Computing of the energy expectation value
    # val = (torch.einsum('ij,ij->',J_i_j_matrix,c_dagger_c_mat) - torch.einsum('i,ii->',chemical_potential, c_dagger_c_mat) 
    #        + 1/4.0* torch.einsum('i,ij,j->',delta_r_tensor,omega_bar_matrix,delta_r_tensor) + 1/4.0*torch.trace(omega_bar_matrix@Gamma_b_tensor)
    #        + torch.einsum('ki,k,i->',delta_gamma_tilde_matrix,delta_r_tensor,torch.diag(c_dagger_c_mat)) 
    #        + 0.5*torch.einsum('ij,ii,jj->',Ve_i_J_matrix,c_dagger_c_mat,c_dagger_c_mat)
    #        - 0.5*torch.einsum('ij,ij,ji->',Ve_i_J_matrix,c_dagger_c_mat,c_dagger_c_mat)
    #        + 0.5*torch.einsum('ij,ij,ji->',Ve_i_J_matrix,c_dagger_c_dagger_mat,c_c_mat)
    #        - 1/4.0 *torch.sum(torch.diag(omega_bar_matrix))     # This is the constant energy shift term. However, it becomes essential to get this correct for the phase shift and hence check this once again
    #        )
    val = phonon_energy + electron_kinetic_energy + electron_electron_interaction_energy + electron_phonon_interaction_energy

    if(val.shape == (1,)):
        raise Exception("The energy expectation value is not a single value. Please check the computation of the energy expectation value.")
    
    return(val)


##############################################################################
##############################################################################
# Bosoic average derivative matrix

def h_delta(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,
            input_variables:cdf.input_variables,
            computed_variables:cdf.computed_variables,
            correlation_matrices:cf.correlation_functions)->torch.Tensor:
    """
    Returns:
    Type : complex torch tensor
    Size : (2N_b,)
    """    
    c_dagger_c_expectation_value_mat = correlation_matrices.c_dagger_c_mat

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

def h_b(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,
        input_variables:cdf.input_variables,
        computed_variables:cdf.computed_variables,
        correlation_matrices:cf.correlation_functions)->torch.Tensor:
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
    c_dagger_c_expectation_value_mat = correlation_matrices.c_dagger_c_mat
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
    
    final_mat = omega_bar_mat - 2*torch.einsum("ij,ij,ijk,ijl->kl", J_i_j_mat, c_dagger_c_expectation_value_mat, alpha_bar_matrix, alpha_bar_matrix)

    return(final_mat)

##############################################################################
##############################################################################
# Functions which compute the partial derivatives of the Energy functional

def epsilon_i_j(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,
                input_variables:cdf.input_variables,
                computed_variables:cdf.computed_variables,
                correlation_matrices:cf.correlation_functions)->torch.Tensor:

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
    c_dagger_c_expectation_value_mat = correlation_matrices.c_dagger_c_mat

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

def Delta_for_h_m(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,
                  input_variables:cdf.input_variables,
                  computed_variables:cdf.computed_variables,
                  correlation_matrices:cf.correlation_functions)->torch.Tensor:
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
    c_c_expectation_value_mat = correlation_matrices.c_c_mat
    
    # # Remember that since Delta = V^{e}_{i,j} *<c_{j}c_{i}> = -V^{e}_{i,j} *<c_{i}c_{j}>  

    # Remark : Just check the sign here once again. Not sure of the sign.
    final_mat = torch.einsum('ij,ji->ij',Ve_real_matrix,c_c_expectation_value_mat)  
    return(final_mat)    

##############################################################################
##############################################################################
# Fermionic covariance derivative matrix
def h_m(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,
        input_variables:cdf.input_variables,
        computed_variables:cdf.computed_variables,
        correlation_matrices:cf.correlation_functions)->torch.Tensor:
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
    epsilon_i_j_mat = epsilon_i_j(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,correlation_matrices)
    Delta_mat_for_h_m = Delta_for_h_m(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,correlation_matrices)

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
                              computed_variables:cdf.computed_variables,correlation_matrices:cf.correlation_functions)->torch.Tensor:
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
    sigma_mat = gcs.sigma(N_b,dtype="complex") 
    denisty_density_anticommutator_connected_correlation_mat = correlation_matrices.density_density_anticommutator_connected_correlation_mat
 
    delta_gamma_tilda_mat = computed_variables.delta_gamma_tilde_mat
    alpha_bar_mat = computed_variables.alpha_bar_mat
    J_i_j_mat = computed_variables.J_i_j_mat
    c_dagger_c_expectation_value_mat = correlation_matrices.c_dagger_c_mat

    # if(correlation_matrix is None):
    #     correlation_mat = cf.correlation_matrix_creation(Gamma_m,N_f)
    #     if(correlation_mat.dtype != torch.complex128):
    #         raise Exception("correlation matri has wrong dtype")
            
    # if(type(correlation_matrix) is np.ndarray):
    #     correlation_mat = torch.from_numpy(correlation_matrix)

    # if(type(correlation_matrix) is torch.Tensor):
    #     correlation_mat = correlation_matrix
    correlation_mat = correlation_matrices.correlation_mat_for_non_gaussian_parameters
   
    # Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
    # Computing the values of the return matrix
    rhs_mat_3 = torch.einsum('kl,li,ij->kj',Gamma_b,delta_gamma_tilda_mat,correlation_mat )
    # print("Computed rhs_mat_3 successfully")

#   Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
    rhs_mat_2 = torch.einsum('kl,jil,ji->kj',sigma_mat,alpha_bar_mat,torch.real(J_i_j_mat*c_dagger_c_expectation_value_mat).to(dtype=torch.complex128) )
    # print("Computed rhs_mat_2 successfully")

#   Check these in the Fourier transform code, to figure out if the equation is correct. Compare with the real space code as well.
    rhs_mat_1 = torch.einsum('kl,iml,im,jim->kj',Gamma_b,alpha_bar_mat,J_i_j_mat,denisty_density_anticommutator_connected_correlation_mat)
    # print("Computed rhs_mat_1 successfully")

    # Combining all of the above matrices into one final matrix which is then transposed and sent back to the cpu to be converted into
    # a numpy array
    final_mat = torch.transpose((-0.5*1j*rhs_mat_1 + rhs_mat_2 + rhs_mat_3),0,1)

    if( torch.any(torch.isinf(final_mat)) or torch.any(torch.isnan(final_mat))  ):
        raise Exception("We get a NaN or an Inf in the final matrix. Please check the computation of the matrix.")
    
    return(final_mat[:,0:N_b])

##############################################################################
##############################################################################
# Functions needed for the computation of the Equation of motion of the Variational parameters (Real time evolution)
def rhs_var_par_eqn_of_motion_lambda_real_time(delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,
                                               input_variables:cdf.input_variables,computed_variables:cdf.computed_variables,
                                               correlation_matrices:cf.correlation_functions)->torch.Tensor:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    Computes the right-hand side of the variational partial differential equation of motion for kappa_1 (and 
    kappa_2 as well).
    -----------------------------

    -----------------------------
    -----------------------------
    Returns:
    -----------------------------
    -----------------------------
    ndarray: The computed matrix.
    TYPE: complex numpy array
    SIZE: N_f x 2*N_b 
    DESCRIPTION:

    """
    # Defining number of the fermionic and bosonic particles     
    N_b = input_variables.N_b

    # Sending all of the tensors to the GPU is it exists, else to the CPU
    sigma_mat = gcs.sigma(N_b,dtype="complex") 
    denisty_density_anticommutator_connected_correlation_mat = correlation_matrices.density_density_anticommutator_connected_correlation_mat
 
    delta_gamma_tilda_mat = computed_variables.delta_gamma_tilde_mat
    alpha_bar_mat = computed_variables.alpha_bar_mat
    J_i_j_mat = computed_variables.J_i_j_mat
    c_dagger_c_expectation_value_mat = correlation_matrices.c_dagger_c_mat
    correlation_mat = correlation_matrices.correlation_mat_for_non_gaussian_parameters
 
    # Computing the term with \delta \tilde{g} in it
    rhs_mat_3 = torch.einsum('kl,li,ij->kj',sigma_mat,delta_gamma_tilda_mat,correlation_mat )

    # Computing the term with Real[ J_j_i <c^{\dagger}_j c_{i}> ]
    rhs_mat_2 = torch.einsum('kl,jil,ji->kj',Gamma_b,alpha_bar_mat,torch.real(J_i_j_mat*c_dagger_c_expectation_value_mat).to(dtype=torch.complex128) )

    # Computing the term with (\alpha_{i,j})_l J_{j,i} <{n_{\alpha},c^{\dagger}_{i} c_{j}}>_{c}
    rhs_mat_1 = torch.einsum('kl,iml,im,jim->kj',sigma_mat,alpha_bar_mat,J_i_j_mat,denisty_density_anticommutator_connected_correlation_mat)

    # Combining all of the above matrices into one final matrix which is then transposed and sent back to the cpu to be converted into
    # a numpy array
    # final_mat = torch.transpose((-0.5*1j*rhs_mat_1 + rhs_mat_2 + rhs_mat_3),0,1)
    final_mat = torch.transpose((0.5*1j*rhs_mat_1 + rhs_mat_2 - rhs_mat_3),0,1)

    if( torch.any(torch.isinf(final_mat)) or torch.any(torch.isnan(final_mat))  ):
        raise Exception("We get a NaN or an Inf in the final matrix. Please check the computation of the matrix.")
    
    if(torch.any(final_mat[:,N_b:].real>1e-10)):
        print(" Warning: The lambda_x part is non zero. Max Difference is : ", torch.max(final_mat[:,N_b:].real))
        # if(torch.any(final_mat[:,N_b:].real>1e-4)):
        #     raise Exception(" The real part of the lambda_x part is too large. Please check the computation of the matrix.")
    # Note that I don't have an equation of motion for \lambda^{x}_{l,m\sigma} (which it might have)
    # Here I just extract the equation of motion for \lambda^{p}_{l,m7sigma} 
    # print("Maximum value for derivative of lambda_x: ",torch.abs(final_mat[:,N_b:]).max() )
    return(final_mat[:,0:N_b])
##############################################################################
##############################################################################
