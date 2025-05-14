# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 06:40:44 2023

@author: Yash_palan
"""


from scipy.integrate import odeint
import numpy as np

import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())

from Common_codes import global_functions_v3_16_4_24 as gf
import math as mt

# from Common_codes import global_functions_v2 as gf
import real_time_evolution_functions_v2_23_4_24 as rtef


def real_time_evo_model(y, t,J_0,omega,lmbda,gamma,position_array,momentum_array,volume,spin_index_exists=True):
    """
    Description of function
    ----------------------------
    ----------------------------
    This function basically computes the RHS for the equation of motion in
    for the Real time evolution equations of motion given in the 
    writeup titled "summary_of_equations".
        
    ----------------------------
    ----------------------------
    Parameters
    ----------------------------
    ----------------------------
    y : TYPE - numpy vector
        DESCRIPTION - This the input
    
    t : TYPE - float
        DESCRIPTION - This is the time index for the odeint function
        
    lmbda  :  TYPE - numpy array
                 SIZE - 1 x 2N_b
                 DESCRIPTION - It stores the variation of lambda (Lang-Firsov transfomation
                               parameter) with momentum k
    position_array : TYPE - numpy array
                        SIZE - 1 x N_f
                        DESCRIPTION - Stores the position space coordinates
           
    momentum_array  : TYPE - numpy array
                         SIZE - 1 x N_b
                         DESCRIPTION - It stores the variation of momentum k
       
    gamma :   TYPE - numpy array
                 SIZE - 1 x N_b
                 DESCRIPTION - It stores the variation of gamma (electron phonon 
                               coupling constant) with momentum k
       
    omega :   TYPE - numpy array
                 SIZE - 1 x N_b
                 DESCRIPTION - It stores the variation of omega (energy of the 
                               phonons with position) with momentum k
       
    J_0 : TYPE - float
             DESCRIPTION - It stores the J_0 (fermion self energy constant)
             
    volume : TYPE - float
                DESCRIPTION - Stores the Volume of the system
    
    ----------------------------
    ----------------------------
    Returns
    ----------------------------
    ----------------------------
    dydt : TYPE - numpy vector
            SIZE -1 x (2N_b + 2N_b*2N_b + 2N_f*2N_f + 2N_b*N_f)
        DESCRIPTION - Stores the output of the time evolution step for the 
        delta_r, Gamma_b and Gamma_f matrices in a vector (in the above mentioned
        order).

    """
    
    # Defining what N_b and N_f are. 
    #  N_b = N_f for our case but we define these two differently for convenience
    # of notation and having same notation as in the writeup
    N_b = len(momentum_array)
    N_f = len(position_array)
    
    # Extracting the delta_R, Gamma_b and Gamma_f
    delta_R = y[0:2*N_b]
    Gamma_b = y[2*N_b:((2*N_b)*(2*N_b)+2*N_b)]
    Gamma_m = y[(2*N_b)*(2*N_b)+2*N_b:((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b)]
    time_derivative_lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):]

    # Reshaping the flattened Gamma_b array to a 2N_bx2N_b
    Gamma_b = np.reshape(Gamma_b,(2*N_b,2*N_b))    
    
    # Reshaping the flattened Gamma_m array to a 2N_fx2N_f
    Gamma_m = np.reshape(Gamma_m,(2*N_f,2*N_f))

    # Reshaping the flattened lambda_bar array to a 2N_bxN_f array
    lambda_bar = np.reshape(time_derivative_lambda_bar,(2*N_b,N_f))
    # Need to update lmbda from lambda_bar which is in accordance with the RK4 method
    lmbda = np.zeros((1,2*N_b))
    # Need to check if the equation below for conversion of lambda_bar to lmbda is correct
    # Notice that this extracts the averaged value of lmbda from lambda_bar since we have the 
    # same lmbda being used for calculation of multiple lambda_bars. The above can help use reduce
    # the stark variations that would otherwise be seen due to dicretisation of the time evolution
    lmbda = 1/N_f*np.diag(np.matmul(lambda_bar[N_b:,:]+complex(0,1)*lambda_bar[:N_b,:] , 
                                    np.transpose(
                                        np.exp(-1j*np.reshape(momentum_array,(N_b,1))*np.reshape(position_array,(1,N_f))  ) 
                                        )  
                                    )
                        )
    
    # Equation of motion for lambda_bar
    time_derivative_lambda_bar = rtef.equation_of_motion_for_Non_Gaussian_parameter(delta_R,Gamma_b,Gamma_m,J_0,N_b,N_f,
                                                                                    gamma,lmbda,omega,momentum_array,
                                                                                    position_array,volume)
     # Equation of motion for delta_R
    d_delta_R_dt = rtef.equation_of_motion_for_bosonic_averages(delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,
                                                                 N_b,N_f,position_array,momentum_array
                                                                ,volume,lmbda,J_0,gamma,omega)
                    
    # Equation of motion for Gamma_b
    d_Gamma_b_dt = rtef.equation_of_motion_for_bosonic_covariance(delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,
                                                                 N_b,N_f,position_array,momentum_array,J_0,lmbda,
                                                                 omega,volume)  
    
    # Equation of motion for Gamma_f
    d_Gamma_f_dt = rtef.equation_of_motion_for_fermionic_covariance(delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,
                                                                    N_b,N_f,position_array,momentum_array,J_0,lmbda,gamma,
                                                                    omega,volume)    
    
   # Reshaping the arrays to a single vector
    d_delta_R_dt = np.reshape(d_delta_R_dt,2*N_b)
    d_Gamma_b_dt = np.reshape(d_Gamma_b_dt,2*N_b*2*N_b)
    d_Gamma_f_dt = np.reshape(d_Gamma_f_dt,2*N_f*2*N_f)
    time_derivative_lambda_bar = np.reshape(time_derivative_lambda_bar,2*N_b*N_f)
    
    # Creating the final array to be returned 
    dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_f_dt))

    return dydt

##############################################################################
##############################################################################
# Checking the working of the functions
# Code works. Run time errors solved. Yash 14/5/24 16:57
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
      Time_derivative_lambda_bar = np.linspace(1,2*N_b*N_f,2*N_b*N_f,endpoint=False).reshape(2*N_b,N_f)
      gamma_m = np.linspace(1,2*N_f*2*N_f,2*N_f*2*N_f).reshape(2*N_f,2*N_f)
      Delta_R = np.ones(2*N_b)
      gamma_b = np.eye(2*N_b)
      J_0 = 2.0
      y = np.concatenate((Delta_R.flatten(),gamma_b.flatten(),gamma_m.flatten(),Time_derivative_lambda_bar.flatten()))
      # Working properly. No run time errors encountered. Yash 11/5/24
      # bosonic_averages_mat = equation_of_motion_for_bosonic_averages(Delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,
      #                                       position_space_grid,momentum_space_grid,volume,lmbda,J_0,gamma,omega)
      
      # Working properly. All run time errors have been removed. Yash 11/5/24 14:02
      # bosonic_covariance_mat = equation_of_motion_for_bosonic_covariance(Delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,
      #                                         position_space_grid,momentum_space_grid,J_0,lmbda,omega,volume)
      
      # Working properly. All run time errors have been removed. Yash 14/5/24 13:51
      # fermionic_covariance_mat = equation_of_motion_for_fermionic_covariance(Delta_R,Gamma_b,Gamma_m,time_derivative_lambda_bar,N_b,N_f,
      #                                           position_space_grid,momentum_space_grid,J_0,lmbda,gamma,omega,volume)
      
      # Working properly. All run time errors have been removed. Yash 14/5/24 15:45
      time_derivative_mat = real_time_evo_model(y,1,J_0,omega,lmbda,gamma,position_space_grid,momentum_space_grid,volume)
      print(time_derivative_mat)

