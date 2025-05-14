# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:43:50 2023

@author: Yash_palan

This file is used for definining all the functions that we will need for Imaginary time evolution
of the state.

"""
##############################################################################
##############################################################################
from Common_codes import global_functions_v3_16_4_24 as gf
import numpy as np
import math as mt
import itertools as it
##############################################################################
##############################################################################
# First equation of motion functions

def equation_of_motion_for_Non_Gaussian_parameter(delta_r,Gamma_b,Gamma_m,J_0,N_b,N_f,
                                                       gamma,lmbda,omega,momentum_array,
                                                       position_array,volume ):
    """
    Calculates the right-hand side (RHS) of the equation of motion for the non-Gaussian parameter.

    Parameters:

    Returns:
    - rhs (ndarray): A matrix representing the RHS of the equation of motion for the non-Gaussian parameter.
    """
#     correlation_mat_for_non_gaussian_parameters = gf.correlation_matrix_creation(Gamma_m,N_f)
        
#     kappa_1_mat = gf.rhs_var_par_eqn_of_motion_kappa_1(delta_r,Gamma_b,Gamma_m,J_0,N_b,N_f,
#                                                        gamma,lmbda,omega,momentum_array,
#                                                        position_array,volume)
    
#     kappa_2_mat = gf.rhs_var_par_eqn_of_motion_kappa_2(delta_r,Gamma_b,Gamma_m,J_0,N_b,N_f,
#                                                        gamma,lmbda,omega,momentum_array,
#                                                        position_array,volume)
    
    time_derivative_lambda_bar = np.zeros((2*N_b,N_f))
    return(time_derivative_lambda_bar)


def equation_of_motion_for_bosonic_averages(delta_r,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,position_array,momentum_array
                    ,volume,lmbda,J_0,gamma,omega):

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 

    Parameters
    ----------
    
    delta_R : TYPE:
              SIZE:
              DESCRIPTION.
              
    Gamma_b : TYPE:
                  
              DESCRIPTION.
              
    Gamma_f : TYPE:
              SIZE:
              DESCRIPTION.
              
    N_b : TYPE:
          SIZE:
          DESCRIPTION.
          
    lmbda : TYPE:
            SIZE:
            DESCRIPTION.
            
    momentum_array : TYPE:
                     SIZE:
                     DESCRIPTION.
                     
    position_array : TYPE:
                     SIZE:
                     DESCRIPTION.
                     
    volume : TYPE:
             SIZE:
             DESCRIPTION.
             
    N_f : TYPE:
          SIZE:
          DESCRIPTION.
          
    J_0 : TYPE:
          SIZE:
          DESCRIPTION.

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 1 x 2N_b
    """
    sigma = np.kron([[0,1],[-1,0]],np.eye(N_b))

    h_delta_matrix = gf.h_delta(delta_r,Gamma_b,Gamma_m,J_0,N_b,N_f,gamma,lmbda,omega,momentum_array,position_array,volume)
    O_delta_mat = gf.O_delta(time_derivative_lambda_bar,Gamma_m,N_b,N_f,volume)
    
    final_mat = np.matmul(sigma, h_delta_matrix) - complex(0,1)*np.matmul(sigma,O_delta_mat)
    return final_mat

def equation_of_motion_for_bosonic_covariance(delta_r,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,
                                              position_array,momentum_array,J_0,lmbda,omega,volume):

    """
    This function calculates the RHS of the equation of motion for the bosonic averages. The point of this is to 
    match this to the review by Shi et al. 

    Parameters
    ----------
    
    delta_R : TYPE:
              SIZE:
              DESCRIPTION.
              
    Gamma_b : TYPE:
                  
              DESCRIPTION.
              
    Gamma_f : TYPE:
              SIZE:
              DESCRIPTION.
              
    N_b : TYPE:
          SIZE:
          DESCRIPTION.
          
    lmbda : TYPE:
            SIZE:
            DESCRIPTION.
            
    momentum_array : TYPE:
                     SIZE:
                     DESCRIPTION.
                     
    position_array : TYPE:
                     SIZE:
                     DESCRIPTION.
                     
    volume : TYPE:
             SIZE:
             DESCRIPTION.
             
    N_f : TYPE:
          SIZE:
          DESCRIPTION.
          
    J_0 : TYPE:
          SIZE:
          DESCRIPTION.

    Returns
    -------
    final_mat : TYPE: complex numpy array
                SIZE: 2N_b x 2N_b
    """

    final_mat = np.zeros((2*N_b,),"complex")    
    sigma = np.kron([[0,1],[-1,0]],np.eye(N_b))
    h_b_matrix = gf.h_b(delta_r,Gamma_b,Gamma_m,J_0,omega,N_b,N_f,lmbda,momentum_array,position_array,volume)
    O_b_matrix = gf.O_b(time_derivative_lambda_bar,N_b)
    final_mat = (np.matmul(sigma,np.matmul(h_b_matrix-complex(0,1)*O_b_matrix,Gamma_b)) \
                    -np.matmul(Gamma_b,np.matmul(h_b_matrix-complex(0,1)*O_b_matrix,sigma)) \
                    )
    return final_mat

def equation_of_motion_for_fermionic_covariance(delta_r,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,
                                                position_array,momentum_array,J_0,lmbda,gamma,omega,volume):

    """
    
    """
    
    sigma = np.kron([[0,1],[-1,0]],np.eye(N_b))
    h_m_matrix = gf.h_m(delta_r,Gamma_b,Gamma_m,J_0,N_b,N_f,gamma,omega,lmbda,momentum_array,position_array,volume)
    O_m_matrix = gf.O_m(time_derivative_lambda_bar,delta_r,N_b,N_f,volume)
    
    final_mat = ( np.matmul(h_m_matrix-complex(0,1)*O_m_matrix, Gamma_m) \
                 -np.matmul( Gamma_m,h_m_matrix-complex(0,1)*O_m_matrix)
                )
    
    return(final_mat)

##############################################################################
##############################################################################
# Checking the working of the functions
  
if __name__=="__main__":    
      number_of_points = 10

      positon_value_max = 10
      positon_value_min = 0
      position_space_grid = gf.coordinate_array_creator_function(0,1,10,True)
      
      momentum_value_max = mt.pi/(positon_value_max-positon_value_min)*number_of_points
      momentum_value_min = -mt.pi/(positon_value_max-positon_value_min)*number_of_points
      momentum_space_grid = gf.coordinate_array_creator_function(0,1,10,False)
      
      N_b = np.size(momentum_space_grid)
      N_f = np.size(position_space_grid)

      lmbda = np.ones(N_b)
      omega = 0.1*np.ones(N_b)
      gamma = np.ones(N_b)
      volume = 1  
      time_derivative_lambda_bar = np.linspace(1,2*N_b*N_f,2*N_b*N_f,endpoint=False).reshape(2*N_b,N_f)
      Gamma_m = np.linspace(1,2*N_f*2*N_f,2*N_f*2*N_f).reshape(2*N_f,2*N_f)
      Delta_R = np.ones(2*N_b)
      Gamma_b = np.eye(2*N_b)
      J = 2.0
      
      # # Working properly. No run time errors encountered. Yash 14/5/24 13:22
      # bosonic_averages_mat = equation_of_motion_for_bosonic_averages(Delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,
      #                                       position_space_grid,momentum_space_grid,volume,lmbda,J_0,gamma,omega)
      
      # Working properly. All run time errors have been removed. Yash 14/5/24 13:55
      # bosonic_covariance_mat = equation_of_motion_for_bosonic_covariance(Delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,
      #                                         position_space_grid,momentum_space_grid,J,lmbda,omega,volume)
      
      # # Working properly. All run time errors have been removed. Yash 14/5/24 15:12
      # fermionic_covariance_mat = equation_of_motion_for_fermionic_covariance(Delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,
      #                                           position_space_grid,momentum_space_grid,J,lmbda,gamma,omega,volume)
      
      time_derivative_mat = equation_of_motion_for_Non_Gaussian_parameter(Delta_R,Gamma_b,Gamma_m,J,N_b,N_f,
                                                       gamma,lmbda,omega,momentum_space_grid,
                                                       position_space_grid,volume )
      print(time_derivative_mat)

      
