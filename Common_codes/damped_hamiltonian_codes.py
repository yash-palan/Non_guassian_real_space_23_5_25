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
import numpy as np
import torch
from Common_codes import generic_codes_20_3_25 as gcs
from Common_codes import class_defn_file_20_3_25 as cdf
from Common_codes import correlation_functions_file_20_3_25 as cf
##############################################################################
##############################################################################

class damped_hamiltonian():
    def __init__(self,phonon_damping:torch.Tensor,fermionic_correlations:cf.correlation_functions,
                 N_b:int,N_f:int):
        self.N_b = N_b
        self.N_f = N_f
        self.phonon_damping = phonon_damping
        self.fermionic_correlations = fermionic_correlations 
        self.Ve_ij_damped_mat = torch.zeros((N_f,N_f),dtype = torch.complex128)
        self.phonon_damping_extended_matrix = torch.diag(torch.concat((torch.diag(phonon_damping),torch.diag(phonon_damping)),dim =0))
        if(self.phonon_damping_extended_matrix.shape !=(2*N_b,2*N_b)):
            raise Exception("Exception: Problem in the shape of the extended damping phonon matrix.")

    def h_delta_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        phonon_damping_mat = self.phonon_damping
        phonon_damping_extended_mat = self.phonon_damping_extended_matrix
        final_mat= (-0.5*torch.einsum('kl,l->k',phonon_damping_extended_mat+phonon_damping_extended_mat.T,delta_r) 
                    +2.0*torch.cat( (torch.einsum('kl,li->k',phonon_damping_mat,lmbda),torch.zeros(self.N_b,dtype=torch.complex128)),dim = 0)
                    )
        return(final_mat)

    def h_b_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        final_mat = -0.5*(self.phonon_damping_extended_matrix+self.phonon_damping_extended_matrix.T)
        return(final_mat)
    
    def Ve_ij_damped_creation(self,lmbda):
        self.Ve_ij_damped_mat =torch.einsum('ki,kl,li->i',lmbda,self.phonon_damping,lmbda)
        return(self.Ve_ij_damped_mat)
    
    def chemical_potentia_damped(self,delta_r,lmbda):
        phonon_damping_mat = self.phonon_damping
        c_dagger_c_mat = self.fermionic_correlations.c_dagger_c_mat
        Ve_ij = self.Ve_ij_damped_mat
        final_mat = (-torch.einsum('kl,li,k->i',phonon_damping_mat,lmbda,delta_r[0:self.N_b])
                    + torch.diag(Ve_ij)
                    + torch.einsum('ij,j->i',Ve_ij,torch.diag(c_dagger_c_mat))
                    )
        return(final_mat)


    def epsilon_i_j_damped(self,delta_r,lmbda):
        chemical_potential_mat = self.chemical_potentia_damped(delta_r,lmbda)
        # Remark- here Ve_ij is defined without the factor of 2.
        Ve_ij = self.Ve_ij_damped_mat
        final_mat = (-torch.einsum('ij,ij->ij',Ve_ij,self.fermionic_correlations.c_dagger_c_mat)
                    + torch.diag(chemical_potential_mat)
                    )
        
        return(final_mat)
    
    def Delta_ij_damped(self):
        final_mat = torch.einsum('nm,mn->nm',self.Ve_ij_damped_mat,self.fermionic_correlations.c_c_mat)
        return(final_mat)
    
    
    def h_m_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        epsilon_ij_mat = self.epsilon_i_j_damped(delta_r,lmbda)
        delta_ij_mat = self.Delta_ij_damped()
        delta_ij_hermitian_conjugate = delta_ij_mat.conj().T

        term_1 = -0.5*(torch.kron(torch.tensor([[-1j,1],[-1,-1j]]), epsilon_ij_mat  ) 
                  + torch.kron(torch.tensor([[1j,1],[-1,1j]]), epsilon_ij_mat.T.contiguous()  ))    
        term_2 = -0.25*(torch.kron(torch.tensor([[-1j,-1],[-1,1j]]), delta_ij_mat  ) 
                  + torch.kron(torch.tensor([[1j,1],[1,-1j]]), delta_ij_mat.T.contiguous()  ))
        term_3 = -0.25*(torch.kron(torch.tensor([[-1j,-1],[-1,1j]]), delta_ij_hermitian_conjugate  ) 
                  + torch.kron(torch.tensor([[1j,1],[1,-1j]]), delta_ij_hermitian_conjugate.T.contiguous()  ))
        final_mat = term_1 + term_2+ term_3
        return(final_mat)
    

    def initialise_and_extract_data(self,delta_r,Gamma_b,Gamma_m,lmbda):
        self.Ve_ij_damped_creation(lmbda)
        self.h_delta_damped_mat = self.h_delta_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        self.h_b_damped_mat = self.h_b_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        self.h_m_damped_mat = self.h_m_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        return
    