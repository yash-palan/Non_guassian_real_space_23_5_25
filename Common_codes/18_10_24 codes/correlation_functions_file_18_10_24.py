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
from Common_codes.class_defn_file_18_10_24 import *
import csv
from Common_codes.generic_codes_18_10_24 import coordinate_array_creator_function
##############################################################################
##############################################################################
# Basic correlators 

def c_dagger_c_expectation_value_matrix_creation(Gamma_m:np.ndarray,N_f:int)->torch.Tensor:
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
    PARAMETERS
    -----------------------------
    -----------------------------
    Gamma_m : TYPE: complex pytorch array
            SIZE: 2N_f x 2N_f
            DESCRIPTION: This is the matrix that is used in the Hamiltonian.    
    N_f : TYPE: int
            DESCRIPTION: Number of fermions in the system. 
    -----------------------------
    -----------------------------    
    RETURNS
    -----------------------------
    -----------------------------
    Type : complex pytorch tensor
    Size : N_f x N_f
 
    """
    # Remark: Yash : 6/9/24 : This expression is correct
    return(
        torch.from_numpy( 
            0.25*(2*np.eye(N_f) - 1j*(Gamma_m[0:N_f,0:N_f] + Gamma_m[N_f:,N_f:] ) + Gamma_m[0:N_f,N_f:] - Gamma_m[N_f:,0:N_f] ) 
                ).to(dtype=torch.complex128)
        )
    
def c_c_dagger_expectation_value_matrix_creation(Gamma_m:np.ndarray,N_f)->torch.Tensor:
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
    PARAMETERS
    -----------------------------
    -----------------------------
    Gamma_m : TYPE: complex pytorch array
            SIZE: 2N_f x 2N_f
            DESCRIPTION: This is the matrix that is used in the Hamiltonian.    
    N_f : TYPE: int
            DESCRIPTION: Number of fermions in the system. 
    -----------------------------
    -----------------------------    
    RETURNS
    -----------------------------
    -----------------------------
    Type : complex pytorch tensor
    Size : N_f x N_f
    """
    # Remark: Yash : 6/9/24 : This expression is correct
    return( 
        torch.from_numpy(
            0.25*(2*np.eye(N_f) - 1j*(Gamma_m[0:N_f,0:N_f] + Gamma_m[N_f:,N_f:] ) - Gamma_m[0:N_f,N_f:] + Gamma_m[N_f:,0:N_f] ) 
                ).to(dtype=torch.complex128)
        )

def c_c_expectation_value_matrix_creation(Gamma_m:np.ndarray,N_f:int)->torch.Tensor:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    This function computes the correlation function <c^_{i}c_{j}>_{GS}.
    This is needed for the computation of the equation of motion of the variational parameters.

    -----------------------------
    -----------------------------
    PARAMETERS
    -----------------------------
    -----------------------------
    Gamma_m : TYPE: complex pytorch array
            SIZE: 2N_f x 2N_f
            DESCRIPTION: This is the matrix that is used in the Hamiltonian.    
    N_f : TYPE: int
            DESCRIPTION: Number of fermions in the system. 
    -----------------------------
    -----------------------------    
    RETURNS
    -----------------------------
    -----------------------------
    Type : complex pytorch tensor
    Size : N_f x N_f
    """
    # Remark: Yash : 6/9/24 : This expression is correct
    return (
        torch.from_numpy(
            0.25*(-1j*(Gamma_m[0:N_f,0:N_f] - Gamma_m[N_f:,N_f:] ) + (Gamma_m[0:N_f,N_f:] + Gamma_m[N_f:,0:N_f]) ) 
            ).to(dtype=torch.complex128)
        ) 

def c_dagger_c_dagger_expectation_value_matrix_creation(Gamma_m:np.ndarray,N_f:int)->torch.Tensor:
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
    PARAMETERS
    -----------------------------
    -----------------------------
    Gamma_m : TYPE: complex pytorch tensor array
            SIZE: 2N_f x 2N_f
            DESCRIPTION: This is the matrix that is used in the Hamiltonian.    
    N_f : TYPE: int
            DESCRIPTION: Number of fermions in the system.     
    -----------------------------
    -----------------------------    
    RETURNS
    -----------------------------
    -----------------------------
    Type : complex pytorch tensor
    Size : N_f x N_f
    """
    # Remark: Yash : 6/9/24 : This expression is correct
    return(
        torch.from_numpy( 
            0.25*(-1j*(Gamma_m[0:N_f,0:N_f] - Gamma_m[N_f:,N_f:] ) - (Gamma_m[0:N_f,N_f:] + Gamma_m[N_f:,0:N_f])  
                  ) 
                ).to(dtype=torch.complex128) 
        )

##############################################################################
##############################################################################
# Functions needed to compute the connected correlation functions
# Remark : Yash : 13/9/24 : I am not checking this for now since we atleast want the Gaussian parts to work properly
# Remark : Yash : 30/9/24 : Need to check these functions

def density_density_anticommutator_connected_correlation_matrix_creation(Gamma_m:np.ndarray,N_f:int)->torch.Tensor:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    This function computes the correlation function <n_{i}n_{j}>_{GS}.
    This is needed for the computation of the equation of motion of the variational parameters.

    -----------------------------
    -----------------------------
    PARAMETERS
    -----------------------------
    -----------------------------
    Gamma_m : TYPE:complex pytorch tensor array
            SIZE: 2N_f x 2N_f
            DESCRIPTION: This is the matrix that is used in the Hamiltonian.    
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

def correlation_matrix_creation(Gamma_m:np.ndarray,N_f:int)->np.ndarray:
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
    final_mat.add(-torch.einsum('ji,ji->ji',c_dagger_c_dagger_expectation_value_mat,c_c_expectation_value_mat))
    final_mat.add_(torch.einsum('ji,ji->ji',c_dagger_c_expectation_value_mat,c_c_dagger_expectation_value_mat) )

#     if( True in set(np.isinf(np.reshape(np.array(final_mat),(-1)))) or True in set(np.isnan(np.reshape(np.array(final_mat),(-1))))  ):
#         raise Exception("We get a NaN or an Inf in the final matrix. Please check the computation of the matrix.")
    return(np.array(final_mat)) 

##############################################################################
##############################################################################
# Checking the working of the functions
# if __name__ == "__main__":
#     # Checking the working of the functions
#     number_of_points = 2
#     positon_value_max = [2 , 2]
#     positon_value_min = [0  , 0]
#     position_space_grid = coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points,True)
#     file_name = "position_space_grid.csv"
#     file = open(file_name, 'a')
#     writer = csv.writer(file)
#     writer.writerows(position_space_grid)
#     file.close()
#     print("Position space grid created")

    
#     momentum_value_max = [np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
#     momentum_value_min = [-np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,-np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
#     momentum_space_grid = coordinate_array_creator_function(momentum_value_min,momentum_value_max,number_of_points,False)
#     file_name = "momentum_space_grid.csv"
#     file = open(file_name, 'a')
#     writer = csv.writer(file)
#     writer.writerows(momentum_space_grid)
#     file.close()
#     print("Momentum space grid created")

#     # np.random.seed(0)
#     N_f = position_space_grid.shape[0]
#     N_b = momentum_space_grid.shape[0]
#     J_0 = np.random.rand(N_f,N_f)
#     # gamma = np.random.rand(N_f,N_f)
#     gamma = np.random.rand(N_b)
#     # omega = np.random.rand(N_b,N_b)
#     omega = np.random.rand(N_b) 
#     lmbda = np.random.rand(N_b)

#     # delta_R= np.random.rand(2*N_b).astype("complex")
#     # Gamma_b= np.random.rand(2*N_b,2*N_b).astype("complex")
#     Gamma_m = np.random.rand(2*N_f,2*N_f).astype("complex")

#     file_name = "Gamma_m.csv"
#     file = open(file_name, 'a')
#     writer = csv.writer(file)
#     writer.writerows(Gamma_m)
#     file.close()

#     file_name = "c_dagger_c_matrix_python.csv"
#     file = open(file_name, 'a')
#     writer = csv.writer(file)
#     writer.writerows(c_dagger_c_expectation_value_matrix_creation(Gamma_m=Gamma_m,N_f=N_f).numpy())
#     file.close()

#     file_name = "c_c_dagger_matrix_python.csv"
#     file = open(file_name, 'a')
#     writer = csv.writer(file)
#     writer.writerows(c_c_dagger_expectation_value_matrix_creation(Gamma_m=Gamma_m,N_f=N_f).numpy())
#     file.close()

#     file_name = "c_c_matrix_python.csv"
#     file = open(file_name, 'a')
#     writer = csv.writer(file)
#     writer.writerows(c_c_expectation_value_matrix_creation(Gamma_m=Gamma_m,N_f=N_f).numpy())
#     file.close()

#     file_name = "c_dagger_c_dagger_python.csv"
#     file = open(file_name, 'a')
#     writer = csv.writer(file)
#     writer.writerows(c_dagger_c_dagger_expectation_value_matrix_creation(Gamma_m=Gamma_m,N_f=N_f).numpy())
#     file.close()



#     # file_name = "J_0.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerows(J_0)
#     # file.close()

#     # file_name = "gamma.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerow(gamma)
#     # # writer.writerows(gamma)
#     # file.close()

#     # file_name = "omega.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # # writer.writerows(omega)
#     # writer.writerow(omega)
#     # file.close()

#     # file_name = "lmbda.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # # writer.writerows(lmbda)
#     # writer.writerow(lmbda)
#     # file.close()

#     # file_name = "delta_R.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # # writer.writerows(delta_R)
#     # writer.writerow(delta_R)
#     # file.close()

#     # file_name = "Gamma_b.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerows(Gamma_b)
#     # file.close()

#     # chemical_potential_val = -5.0
#     # # Volume of the system
#     # volume = np.prod(np.array(positon_value_max)-np.array(positon_value_min))
#     # # Defining the object which stores all the input variables
#     # initial_input_variables = input_variables(position_space_grid,momentum_space_grid,lmbda,J_0,gamma,omega,chemical_potential_val)
#     # computed_variables_instance = computed_variables(N_b,N_f)
#     # # print("Completed getting computed variables instance =",t,".")

#     # # Computing the values for the computed_varaibles class
#     # computed_variables_instance.initialize_all_variables(initial_input_variables,delta_R,Gamma_b)

#     # file_name = "omega_bar_mat.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerows(computed_variables_instance.omega_bar_mat.numpy())
#     # file.close()

#     # file_name = "alpha_bar_mat.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerow(np.reshape( computed_variables_instance.alpha_bar_mat.numpy(),-1) )
#     # # writer.writerows(np.reshape( computed_variables_instance.alpha_bar_mat.numpy(),-1) )
#     # file.close()

#     # file_name = "delta_gamma_tilde_mat.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerows(computed_variables_instance.delta_gamma_tilde_mat.numpy())
#     # file.close()

#     # file_name = "Ve_i_j_mat.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerows(computed_variables_instance.Ve_i_j_mat.numpy() )
#     # file.close()

#     # file_name = "chemical_potential.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerow(computed_variables_instance.chemical_potential(initial_input_variables).numpy() )
#     # # writer.writerows(computed_variables_instance.chemical_potential(initial_input_variables).numpy() )
#     # file.close()
    
#     # file_name = "J_i_j_mat.csv"
#     # file = open(file_name, 'a')
#     # writer = csv.writer(file)
#     # writer.writerows(computed_variables_instance.J_i_j_mat.numpy() )
#     # file.close()
#     # print("\n Completed storing all the data in the files")
