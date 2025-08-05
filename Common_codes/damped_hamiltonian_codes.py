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
import torch
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
        self.h_delta_damped_mat = torch.zeros((2*N_b,),dtype=torch.complex128)
        self.h_b_damped_mat = torch.zeros((2*N_b,2*N_b),dtype=torch.complex128)
        self.h_m_damped_mat = torch.zeros((2*N_f,2*N_f),dtype=torch.complex128)
        if(self.phonon_damping_extended_matrix.shape !=(2*N_b,2*N_b)):
            raise Exception("Exception: Problem in the shape of the extended damping phonon matrix.")
        

class damped_hamiltonian_model_2():
    def __init__(self,phonon_damping:torch.Tensor,fermionic_correlations:cf.correlation_functions,
                 N_b:int,N_f:int):
        # super().__init__(phonon_damping,fermionic_correlations,N_b,N_f)
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
        c_dagger_c_diag = torch.diag(self.fermionic_correlations.c_dagger_c_mat)
        final_mat= (-0.5*torch.einsum('kl,l->k',phonon_damping_extended_mat+phonon_damping_extended_mat.T,delta_r) 
                    +2.0*torch.cat( (torch.einsum('kl,li,i->k',phonon_damping_mat,lmbda,c_dagger_c_diag),torch.zeros(self.N_b,dtype=torch.complex128)),dim = 0)
                    )
        return(final_mat)

    def h_b_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        final_mat = -0.5*(self.phonon_damping_extended_matrix+self.phonon_damping_extended_matrix.T)
        return(final_mat)
    
    def Ve_ij_damped_creation(self,lmbda):
        # self.Ve_ij_damped_mat =torch.einsum('ki,kl,lj->ij',lmbda,self.phonon_damping,lmbda)
        self.Ve_ij_damped_mat = 2*torch.einsum('ki,kl,lj->ij',lmbda,self.phonon_damping,lmbda)
        return(self.Ve_ij_damped_mat)
    
    def chemical_potential_damped(self,delta_r,lmbda):
        phonon_damping_mat = self.phonon_damping
        c_dagger_c_mat = self.fermionic_correlations.c_dagger_c_mat
        Ve_ij = self.Ve_ij_damped_mat
        final_mat = (-torch.einsum('kl,li,k->i',phonon_damping_mat,lmbda,delta_r[0:self.N_b])
                    + torch.diag(Ve_ij)/2
                    + torch.einsum('ij,j->i',Ve_ij,torch.diag(c_dagger_c_mat))
                    )
        return(final_mat)


    def epsilon_i_j_damped(self,delta_r,lmbda):
        chemical_potential_mat = self.chemical_potential_damped(delta_r,lmbda)
        Ve_ij = self.Ve_ij_damped_mat
        # final_mat = (-torch.einsum('ij,ij->ij',Ve_ij,self.fermionic_correlations.c_dagger_c_mat)
        #             + torch.diag(chemical_potential_mat)
        #             )
        final_mat = (-torch.einsum('ij,ji->ij',Ve_ij,self.fermionic_correlations.c_dagger_c_mat)
                    + torch.diag(chemical_potential_mat)
                    )
        
        return(final_mat)
    
    def Delta_ij_damped(self):
        # final_mat = torch.einsum('nm,mn->nm',self.Ve_ij_damped_mat,self.fermionic_correlations.c_c_mat)
        final_mat = torch.einsum('ij,ji->ij',self.Ve_ij_damped_mat,self.fermionic_correlations.c_c_mat)

        return(final_mat)
    
    
    def h_m_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        epsilon_ij_mat = self.epsilon_i_j_damped(delta_r,lmbda)
        delta_ij_mat = self.Delta_ij_damped()
        delta_ij_hermitian_conjugate = delta_ij_mat.conj().T.contiguous()

        term_1 = -0.5*(torch.kron(torch.tensor([[-1j,1],[-1,-1j]]), epsilon_ij_mat  ) 
                  + torch.kron(torch.tensor([[1j,1],[-1,1j]]), epsilon_ij_mat.T.contiguous()  ))    
        term_2 = -0.25*(torch.kron(torch.tensor([[-1j,-1],[-1,1j]]), delta_ij_mat  ) 
                  + torch.kron(torch.tensor([[1j,1],[1,-1j]]), delta_ij_mat.T.contiguous()  ))
        term_3 = -0.25*(torch.kron(torch.tensor([[-1j,1],[1,1j]]), delta_ij_hermitian_conjugate  ) 
                  + torch.kron(torch.tensor([[1j,-1],[-1,-1j]]), delta_ij_hermitian_conjugate.T.contiguous()  ))
        final_mat_1 = term_1 + term_2+ term_3        

        term_1 = -0.5*(torch.kron(torch.tensor([[-1j,1],[-1,-1j]]), epsilon_ij_mat  ) 
                  + torch.kron(torch.tensor([[1j,1],[-1,1j]]), epsilon_ij_mat.T.contiguous()  ))    
        term_2 = -0.5*(torch.kron(torch.tensor([[-1j,-1],[-1,1j]]), delta_ij_mat  ) )
        term_3 = -0.5*(torch.kron(torch.tensor([[-1j,1],[1,1j]]), delta_ij_hermitian_conjugate  ) )
        # final_mat.add_( 0.5*torch.kron(torch.tensor([  [ -1j, 1  ],[  1 ,  1j ]  ]),Delta_mat_for_h_m.t().conj().contiguous()  )  )  
        final_mat_2 = term_1 + term_2+ term_3        
        if(torch.any(torch.abs(final_mat_1-final_mat_2) > 1e-7) or torch.any(final_mat_2.imag>1e-7) ):
            raise Exception("Exception: Check the calculation for h_m damped")
        
        final_mat = final_mat_1
        return(final_mat)
    
    def initialise_and_extract_data(self,delta_r,Gamma_b,Gamma_m,lmbda):
        self.Ve_ij_damped_creation(lmbda)
        self.h_delta_damped_mat = self.h_delta_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        self.h_b_damped_mat = self.h_b_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        self.h_m_damped_mat = self.h_m_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        return
    
##############################################################################
##############################################################################
class damped_hamiltonian_model_1():
    """
    This is the class for the case when we add the damping to just the final
    """
    def __init__(self,phonon_damping:torch.Tensor,fermionic_correlations:cf.correlation_functions,
                 N_b:int,N_f:int):
        # super().__init__(phonon_damping,fermionic_correlations,N_b,N_f)
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
        final_mat= -0.5*torch.einsum('kl,l->k',phonon_damping_extended_mat+phonon_damping_extended_mat.T,delta_r) 
        return(final_mat)

    def h_b_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        final_mat = -0.5*(self.phonon_damping_extended_matrix+self.phonon_damping_extended_matrix.T)
        return(final_mat) 

    def initialise_and_extract_data(self,delta_r,Gamma_b,Gamma_m,lmbda):
        self.h_delta_damped_mat = self.h_delta_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        self.h_b_damped_mat = self.h_b_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        self.h_m_damped_mat = torch.zeros((2*self.N_f,2*self.N_f),dtype = torch.complex128)
        return

##############################################################################
##############################################################################
class damped_hamiltonian_model_3(damped_hamiltonian):
    """
    For the case when we add damping after the Lang firsov 
    damping hamiltonian = -i\\sum_i \\Gamma_i x_i
    """
    def h_delta_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        phonon_damping_mat = self.phonon_damping
        # phonon_damping_extended_mat = self.phonon_damping_extended_matrix
        phonon_damping_mat_actual = torch.diag(phonon_damping_mat)
        final_mat= -2.0*torch.cat( (phonon_damping_mat_actual,torch.zeros(self.N_b,dtype=torch.complex128)),dim = 0)
        return(final_mat)
    
    def initialise_and_extract_data(self,delta_r,Gamma_b,Gamma_m,lmbda):
        self.h_delta_damped_mat = self.h_delta_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        self.h_b_damped_mat = torch.zeros((2*self.N_b,2*self.N_b),dtype = torch.complex128)
        self.h_m_damped_mat = torch.zeros((2*self.N_f,2*self.N_f),dtype = torch.complex128)
        return   
##############################################################################
##############################################################################
class damped_hamiltonian_model_4(damped_hamiltonian):
    """
    For the case when we add damping before the Lang firsov 
    damping hamiltonian = -i\\sum_i \\Gamma_i x_i
    """
    def h_delta_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        phonon_damping_mat = self.phonon_damping
        # phonon_damping_extended_mat = self.phonon_damping_extended_matrix
        phonon_damping_mat_actual = torch.diag(phonon_damping_mat)
        # phonon_damping_extended_mat = self.phonon_damping_extended_matrix
        final_mat=-2.0*torch.cat( (phonon_damping_mat_actual,torch.zeros(self.N_b,dtype=torch.complex128)),dim = 0)
        return(final_mat)
    
    def epsilon_i_j_damped(self,delta_r,lmbda):
        phonon_damping_mat_actual = torch.diag(self.phonon_damping)
        final_mat = torch.diag(torch.einsum('q,qj->j',phonon_damping_mat_actual,lmbda))
        return(final_mat)

    def h_m_damped(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor):
        epsilon_ij_mat = self.epsilon_i_j_damped(delta_r,lmbda)
        term_1 =(torch.kron(torch.tensor([[-1j,1],[-1,-1j]]), epsilon_ij_mat  ) 
                  + torch.kron(torch.tensor([[1j,1],[-1,1j]]), epsilon_ij_mat.T.contiguous()  ))    
        final_mat = term_1
        return(final_mat)

    def initialise_and_extract_data(self,delta_r:torch.Tensor,Gamma_b:torch.Tensor,Gamma_m:torch.Tensor,lmbda:torch.Tensor)->None:
        self.h_delta_damped_mat = self.h_delta_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        self.h_b_damped_mat = torch.zeros((2*self.N_b,2*self.N_b),dtype = torch.complex128)
        self.h_m_damped_mat = self.h_m_damped(delta_r,Gamma_b,Gamma_m,lmbda)
        return   


##############################################################################
##############################################################################