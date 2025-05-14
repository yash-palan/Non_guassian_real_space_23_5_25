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
from scipy import linalg as la
import torch
from Common_codes import hamiltonian_derivative_matrices_18_10_24 as hdm
from Common_codes import non_gaussian_transform_derivative_18_10_24 as ngtd
from Common_codes import class_defn_file_18_10_24 as cdf
from Common_codes import correlation_functions_file_18_10_24 as cf
##############################################################################
##############################################################################
# First equation of motion functions

# def equation_of_motion_for_Non_Gaussian_parameter(delta_r:np.ndarray,Gamma_b:np.ndarray,Gamma_m:np.ndarray,
#                                                 input_variables:cdf.input_variables,computed_variables:cdf.computed_variables)->np.ndarray:
#       """
#       Calculates the right-hand side (RHS) of the equation of motion for the non-Gaussian parameter.

#       Parameters:
      

#       Returns:
#       - time_derivative_lambda_bar : 
#       TYPE: complex numpy array
#       SIZE: N_b x N_f
#       Matrix which stores the time derivative of the non-Gaussian parameter.
#       """
#       N_f = input_variables.N_f
#       N_b = input_variables.N_b
#       # start_time = time.time()
#       print("\n correlation_mat_for_non_gaussian_parameters = ")
#       correlation_mat_for_non_gaussian_parameters = cf.correlation_matrix_creation(Gamma_m,N_f)
#       print(correlation_mat_for_non_gaussian_parameters)
#       # print("Time required for correlation matrix creation: ",time.time()-start_time)

#       # kappa_1_mat,kappa_2_mat = hdm.rhs_var_par_eqn_of_motion(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,
#       #                                                       correlation_mat_for_non_gaussian_parameters)
#       kappa_1_mat = hdm.rhs_var_par_eqn_of_motion(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,
#                                                             correlation_mat_for_non_gaussian_parameters)
#       # print("Time required for kappa_1 and kappa_2 matrix creation: ",time.time()-start_time)
#       print("\n kappa_1_mat = ")
#       print(kappa_1_mat)

#       print("rcond value for correlation matrix is : ", 1/np.linalg.cond(correlation_mat_for_non_gaussian_parameters))

#       time_derivative_lambda_bar = la.solve(correlation_mat_for_non_gaussian_parameters,kappa_1_mat)
#       # inv_mat = la.inv(correlation_mat_for_non_gaussian_parameters)
#       # print("\n inv_mat = ")
#       # print(inv_mat)

#       # time_derivative_lambda_bar = np.transpose( np.matmul( la.inv(correlation_mat_for_non_gaussian_parameters), 
#       #                         kappa_1_mat ) )
      
#       return(time_derivative_lambda_bar)

def equation_of_motion_for_Non_Gaussian_parameter_spin_modes_summed(delta_r:np.ndarray,Gamma_b:np.ndarray,Gamma_m:np.ndarray,
                                                input_variables:cdf.input_variables,computed_variables:cdf.computed_variables, spin_index=True)->np.ndarray:
      """
      Calculates the right-hand side (RHS) of the equation of motion for the non-Gaussian parameter.

      Parameters:
      

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
      correlation_mat_for_non_gaussian_parameters = cf.correlation_matrix_creation(Gamma_m,N_f)
      np.save("correlation_mat_for_non_gaussian_parameters.npy",correlation_mat_for_non_gaussian_parameters)

      kappa_1_mat = hdm.rhs_var_par_eqn_of_motion(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,
                                                            correlation_mat_for_non_gaussian_parameters)
      np.save("kappa_1_mat.npy",kappa_1_mat)
      if(spin_index==True):
            spin_summed_correlation_mat = (correlation_mat_for_non_gaussian_parameters[0:N_b,0:N_b] + correlation_mat_for_non_gaussian_parameters[0:N_b,N_b:] 
                                          +correlation_mat_for_non_gaussian_parameters[N_b:,0:N_b]  + correlation_mat_for_non_gaussian_parameters[N_b:,N_b:]
                                          )
            spin_summed_kappa_1_mat = kappa_1_mat[0:N_b,:] + kappa_1_mat[N_b:,:]             
      cond_val = np.linalg.cond(correlation_mat_for_non_gaussian_parameters)
      print(" cond value for correlation matrix is : ", cond_val)
      # print("rcond value for correlation matrix is : ", np.linalg.cond(correlation_mat_for_non_gaussian_parameters))

      # time_derivative_lambda_bar = la.solve(correlation_mat_for_non_gaussian_parameters,kappa_1_mat)
      # time_derivative_lambda_bar = la.solve(spin_summed_correlation_mat,spin_summed_kappa_1_mat)      
      # time_derivative_lambda_bar,residues,rank,s = la.lstsq(spin_summed_correlation_mat,spin_summed_kappa_1_mat, cond = cond_val)      
      time_derivative_lambda_bar = la.lstsq(spin_summed_correlation_mat,spin_summed_kappa_1_mat, cond = cond_val)      
      np.save("time_derivative_lambda_bar.npy",time_derivative_lambda_bar[0].T)      # type: ignore

      # raise Exception(time_derivative_lambda_bar)

      # We send back the transpose since time_derivative_lambda_bar is a matrix with N_f/2 x N_b dimensions and not N_b x N_f/2
      # (just check if the above statement is true)
      return(time_derivative_lambda_bar[0].T) # type: ignore

def equation_of_motion_for_Non_Gaussian_parameter_lambda(delta_r:np.ndarray,Gamma_b:np.ndarray,Gamma_m:np.ndarray,
                                                input_variables:cdf.input_variables,computed_variables:cdf.computed_variables)->np.ndarray:
      """
      Description:
      ------------
      Calculates the right-hand side (RHS) of the equation of motion for the non-Gaussian parameter.
      \\partial_{\\tau} \\lambda_q is computed rather than  \\partial_{\\tau} \\lambda_bar
      
      
      Parameters:
      ------------

      Returns:
      ------------
      time_derivative_lambda_bar : 
      TYPE: complex numpy array
      SIZE: 1 x N_b 
      Matrix which stores the time derivative of the non-Gaussian parameter.
      """
      N_f = input_variables.N_f

      correlation_mat_for_non_gaussian_parameters = cf.correlation_matrix_creation(Gamma_m,N_f)
      denominator = np.einsum('qm,mn->q',np.conj(input_variables.fourier_array),correlation_mat_for_non_gaussian_parameters)

      kappa_1_new_mat = hdm.rhs_var_par_eqn_of_motion_lambda(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables,
                                                            correlation_mat_for_non_gaussian_parameters)

      time_derivative_lambda_bar = kappa_1_new_mat/denominator
      
      return(time_derivative_lambda_bar)


def equation_of_motion_for_bosonic_averages(delta_r:np.ndarray,Gamma_b:np.ndarray,Gamma_m:np.ndarray,time_derivative_lambda_bar:np.ndarray,
                                            input_variables:cdf.input_variables,computed_variables:cdf.computed_variables):

      """
      This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
      match this to the review by Shi et al. 

      Parameters
      ----------
      
      delta_R : TYPE: complex numpy array
                  SIZE: 1x 2N_b
                  DESCRIPTION: Stores the average value of the bosonic quadrature operators
                  
      Gamma_b : TYPE: complex numpy array
                  SIZE: 2N_b x 2N_b
                  DESCRIPTION: Stores the covariance matrix of the bosonic operators
                  
      Gamma_f : TYPE: complex numpy array
                  SIZE: 2N_f x 2N_f
                  DESCRIPTION: Stores the covariance matrix of the fermionic operators 
                  
      N_b : TYPE: int
            DESCRIPTION: Stores the number of points in the bosonic grid (basically the number of bosons)

      N_f : TYPE: int
            DESCRIPTION: Stores the number of points in the fermionic grid (basically the number of fermions) 

      lmbda : TYPE: complex numpy array
                  SIZE: 1 x 2N_b
                  DESCRIPTION: Stores the values of the non-Gaussian parameter 
                  
      momentum_array : TYPE: complex numpy array
                        SIZE: N_b x dim
                        DESCRIPTION: Stores the momentum values of the bosonic grid
                        
      position_array : TYPE: complex numpy array
                        SIZE: N_f x dim
                        DESCRIPTION: Stores the position values of the fermion grid
                                          
      J_0 : TYPE: float
            DESCRIPTION: Stores the value of the hopping parameter

      Returns
      -------
      final_mat : TYPE: complex numpy array
                  SIZE: 1 x 2N_b
                  This function returns the term given by eqn 19 in the writeup 
                  "Ch-summary_of_equations" on GITHUB as the matrix
      """
      N_b = input_variables.N_b

      sigma = torch.from_numpy(np.kron([[0,1],[-1,0]],np.eye(N_b))).to(torch.complex128)
      h_delta_matrix = hdm.h_delta(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)
      O_delta_mat = ngtd.O_delta(time_derivative_lambda_bar,Gamma_m,input_variables)

      np.save("h_delta_matrix.npy",h_delta_matrix)
      np.save("O_delta_mat.npy",O_delta_mat)
      # final_mat = -np.matmul(Gamma_b, h_delta_matrix) - complex(0,1)*np.matmul(sigma,O_delta_mat)
      final_mat = -torch.matmul(torch.tensor(Gamma_b).to(dtype=torch.complex128), h_delta_matrix) - complex(0,1)*torch.matmul(sigma,O_delta_mat)

      return(final_mat.numpy())

def equation_of_motion_for_bosonic_covariance(delta_r:np.ndarray,Gamma_b:np.ndarray,Gamma_m:np.ndarray,time_derivative_lambda_bar:np.ndarray,
                                              input_variables:cdf.input_variables,computed_variables:cdf.computed_variables):
      # Working properly. All run time errors have been removed. Yash 11/5/24 14:02
      """
      This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
      match this to the review by Shi et al. 

      Parameters
      ----------
      
      delta_R : TYPE: complex numpy array
                  SIZE: 1x 2N_b
                  DESCRIPTION: Stores the average value of the bosonic quadrature operators
                  
      Gamma_b : TYPE: complex numpy array
                  SIZE: 2N_b x 2N_b
                  DESCRIPTION: Stores the covariance matrix of the bosonic operators
                  
      Gamma_f : TYPE: complex numpy array
                  SIZE: 2N_f x 2N_f
                  DESCRIPTION: Stores the covariance matrix of the fermionic operators 
                  
      N_b : TYPE: int
            DESCRIPTION: Stores the number of points in the bosonic grid (basically the number of bosons)

      N_f : TYPE: int
            DESCRIPTION: Stores the number of points in the fermionic grid (basically the number of fermions) 

      lmbda : TYPE: complex numpy array
                  SIZE: 1 x 2N_b
                  DESCRIPTION: Stores the values of the non-Gaussian parameter 
                  
      momentum_array : TYPE: complex numpy array
                        SIZE: N_b x dim
                        DESCRIPTION: Stores the momentum values of the bosonic grid
                        
      position_array : TYPE: complex numpy array
                        SIZE: N_f x dim
                        DESCRIPTION: Stores the position values of the fermion grid

                  
      J_0 : TYPE: float
            DESCRIPTION: Stores the value of the hopping parameter

      Returns
      -------
      final_mat : TYPE: complex numpy array
                  SIZE: 2N_b x 2N_b
                  This function returns the term given by eqn 19 in the writeup 
                  "Ch-summary_of_equations" on GITHUB as the matrix
      """
      N_b = input_variables.N_b

      #     final_mat = np.zeros((2*N_b,),"complex")    
      sigma = torch.tensor(np.kron([[0,1],[-1,0]],np.eye(N_b))  ,dtype=torch.complex128)
      Gamma_b_tensor = torch.tensor(Gamma_b,dtype=torch.complex128)

      # Remember that these definions are extremely important for the calculation of the h_b matrix and should
      # be passed correctly to the function. Any change in the definition of the h_b function SHOULD be reflected
      # in here as well

      h_b_matrix = hdm.h_b(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)
      np.save("h_b_matrix.npy",h_b_matrix)

      # Remember that these definions are extremely important for the calculation of the O_b matrix and should
      # be passed correctly to the function. Any change in the definition of the O_b function SHOULD be reflected
      # in here as well
      O_b_matrix = ngtd.O_b(time_derivative_lambda_bar,input_variables)
      # final_mat = (np.matmul(np.transpose(sigma),np.matmul(h_b_matrix,sigma)) \
      #                   -np.matmul(Gamma_b,np.matmul(h_b_matrix,Gamma_b)) \
      #                   -complex(0,1)*np.matmul(sigma,np.matmul(O_b_matrix,Gamma_b)) \
      #                   +complex(0,1)*np.matmul(Gamma_b,np.matmul(O_b_matrix,sigma)))

      final_mat = (torch.matmul(sigma.t(),torch.matmul(h_b_matrix,sigma)) \
                              -torch.matmul(Gamma_b_tensor,torch.matmul(h_b_matrix,Gamma_b_tensor)) \
                              -complex(0,1)*torch.matmul(sigma,torch.matmul(O_b_matrix,Gamma_b_tensor)) \
                              +complex(0,1)*torch.matmul(Gamma_b_tensor,torch.matmul(O_b_matrix,sigma)))

      return final_mat

def equation_of_motion_for_fermionic_covariance(delta_r:np.ndarray,Gamma_b:np.ndarray,Gamma_m:np.ndarray,time_derivative_lambda_bar:np.ndarray,
                                                input_variables:cdf.input_variables,computed_variables:cdf.computed_variables):

      """
      This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
      match this to the review by Shi et al. 

      ------------------------------------
      ------------------------------------
      Parameters
      ------------------------------------
      ------------------------------------
      delta_r : TYPE: complex numpy array
                  SIZE: 1x 2N_b
                  DESCRIPTION: Stores the average value of the bosonic quadrature operators
                  
      Gamma_b : TYPE: complex numpy array
                  SIZE: 2N_b x 2N_b
                  DESCRIPTION: Stores the covariance matrix of the bosonic operators
                  
      Gamma_f : TYPE: complex numpy array
                  SIZE: 2N_f x 2N_f
                  DESCRIPTION: Stores the covariance matrix of the fermionic operators 
                  
      N_b : TYPE: int
            DESCRIPTION: Stores the number of points in the bosonic grid (basically the number of bosons)

      N_f : TYPE: int
            DESCRIPTION: Stores the number of points in the fermionic grid (basically the number of fermions) 

      lmbda : TYPE: complex numpy array
                  SIZE: 1 x 2N_b
                  DESCRIPTION: Stores the values of the non-Gaussian parameter 
                  
      momentum_array : TYPE: complex numpy array
                        SIZE: N_b x dim
                        DESCRIPTION: Stores the momentum values of the bosonic grid
                        
      position_array : TYPE: complex numpy array
                        SIZE: N_f x dim
                        DESCRIPTION: Stores the position values of the fermion grid
                  
      J_0 : TYPE: float
            DESCRIPTION: Stores the value of the hopping parameter

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
      # N_b = input_variables.N_b  
      h_m_matrix = hdm.h_m(delta_r,Gamma_b,Gamma_m,input_variables,computed_variables)
      O_m_matrix = ngtd.O_m(time_derivative_lambda_bar,delta_r,input_variables)
      np.save("h_m_matrix.npy",h_m_matrix)
      np.save("O_m_matrix.npy",O_m_matrix)
      # final_mat = ( -h_m_matrix - np.matmul(Gamma_m,np.matmul(h_m_matrix,Gamma_m)) \
      #               + complex(0,1)*(np.matmul(Gamma_m,O_m_matrix)-np.matmul(O_m_matrix,Gamma_m)) \
      #             )
      Gamma_m_tensor = torch.tensor(Gamma_m,dtype=torch.complex128)
      final_mat = ( -h_m_matrix - torch.matmul(Gamma_m_tensor,np.matmul(h_m_matrix,Gamma_m_tensor)) \
                  + complex(0,1)*(np.matmul(Gamma_m_tensor,O_m_matrix)-np.matmul(O_m_matrix,Gamma_m_tensor)) \
            )
      
      return(final_mat)
