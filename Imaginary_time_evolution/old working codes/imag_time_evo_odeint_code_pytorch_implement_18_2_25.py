# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:48:18 2023

@author: Yash_palan

This code contains the code that will
"""

#########################################
#########################################
# import time
import gc
import time
from scipy.integrate import odeint
import numpy as np
from Imaginary_time_evolution import imaginary_time_evolution_functions_pytorch_implement_18_2_25 as itef
from Common_codes import class_defn_file_18_10_24 as cdf
# from Common_codes import file_with_checks as fwc

##############################################################################
##############################################################################

def imag_time_evo_model_solve_ivp(t:np.ndarray,y:np.ndarray,input_variables:cdf.input_variables):
    """
    Parameters
    ----------
    y : TYPE - numpy array 
        DESCRIPTION -
    
    t : TYPE - 
        DESCRIPTION -
            
    
    Returns
    -------
    dydt : TYPE - numpy vector
            SIZE -1 x (2N_b + 2N_b*2N_b + 2N_f*2N_f + 2N_b*2N_f + 2N_b*N_f)
        DESCRIPTION - Stores the output of the time evolution step for the 
        delta_r, Gamma_b, Gamma_f and lambda_bar  matrices 
        in a vector (in the above mentioned order).

    """    
    print(" Started time =",t,".")
    
    # Defining some basic quantities that we use repeatedly in this file N_b and N_f 
    N_b = input_variables.N_b
    N_f = input_variables.N_f
    
    # Extracting the delta_R, Gamma_b and Gamma_f
    if(y.dtype != "complex"):
        delta_R = y[0:2*N_b].astype("complex")
        Gamma_b = y[2*N_b:((2*N_b)*(2*N_b)+2*N_b)].astype("complex")
        Gamma_m = y[(2*N_b)*(2*N_b)+2*N_b:((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b)].astype("complex")
        # lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):].astype("complex")
        lmbda_q = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):].astype("complex")
    
    if(y.dtype == "complex"):
        delta_R = y[0:2*N_b]
        Gamma_b = y[2*N_b:((2*N_b)*(2*N_b)+2*N_b)]
        Gamma_m = y[(2*N_b)*(2*N_b)+2*N_b:((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b)]
        # lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):]
        lmbda_q = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):]
    
    # Reshaping the flattened Gamma_b array to a 2N_bx2N_b
    Gamma_b = np.reshape(Gamma_b,(2*N_b,2*N_b))    
    
    # Reshaping the flattened Gamma_m array to a 2N_fx2N_f
    Gamma_m = np.reshape(Gamma_m,(2*N_f,2*N_f))
    
    # Reshaping the 
    # lmbda_q = np.reshape(lmbda_q,(N_b,))
    # Reshaping the spin_removed_lambda_bar array to a N_b x N_f/2 array

    lmbda_q = np.reshape(lmbda_q,(N_b,int(N_f/2))) # We get N_f/2 since we have spin in this simulation.

    # input_variables.updating_lambda_bar_from_lambda(lmbda_q=lmbda_q,volume=N_b)  # Note that here we need to change the volume if we ever do. Else, it is N_b.
    input_variables.updating_lambda_bar_from_spin_removed_lambda_bar(lmbda_q,spin_index=True,update_in_input_variables=True)  
    
    # lambda_bar = np.reshape(lambda_bar,(2*N_b,N_f))

    # lambda_bar = np.reshape(lambda_bar,(N_b,N_f)) 
    
    # Initialising the computational_variables class
    computed_variables_instance = cdf.computed_variables(N_b,N_f)
    # print("Completed getting computed variables instance =",t,".")

    # Computing the values for the computed_varaibles class
    computed_variables_instance.initialize_all_variables(input_variables,delta_R,Gamma_b)

    # start_time = time.time()
    # Equation of motion for lambda_bar
    # time_derivative_lambda= itef.equation_of_motion_for_Non_Gaussian_parameter_lambda(delta_R,Gamma_b,Gamma_m,
    #                                                                                     input_variables,
    #                                                                                     computed_variables_instance)
    time_derivative_lambda= itef.equation_of_motion_for_Non_Gaussian_parameter_spin_modes_summed(delta_R,Gamma_b,Gamma_m,
                                                                                    input_variables,
                                                                                    computed_variables_instance)
    # time_derivative_lambda = np.zeros((N_b,))
    time_derivative_lambda_bar = input_variables.updating_lambda_bar_from_spin_removed_lambda_bar(time_derivative_lambda,
                                                                                spin_index=True ,update_in_input_variables=False)
    # print("Time taken for lambda_bar = ",time.time()-start_time)
    # if(np.any(np.imag(time_derivative_lambda_bar)>1e-4)):
    #     raise Exception("There is a large imaginary part in the time_derivative_lambda_bar. Check the code and the evolution again.")

    # Equation of motion for delta_R
    # start_time = time.time()
    d_delta_R_dt = itef.equation_of_motion_for_bosonic_averages(delta_R,Gamma_b,Gamma_m,
                                                                time_derivative_lambda_bar,
                                                                input_variables,
                                                                computed_variables_instance)
    # print("Time taken for Delta_R = ",time.time()-start_time)
    # print("dellta_R")
    # Equation of motion for Gamma_b
    # start_time = time.time()
    d_Gamma_b_dt = itef.equation_of_motion_for_bosonic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                    time_derivative_lambda_bar,
                                                                    input_variables,
                                                                    computed_variables_instance)  
    # print("Time taken for Gamma_b = ",time.time()-start_time)
    # print("Gamma_b")
    
    # Equation of motion for Gamma_f
    # start_time = time.time()
    d_Gamma_m_dt = itef.equation_of_motion_for_fermionic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                    time_derivative_lambda_bar,
                                                                    input_variables,
                                                                    computed_variables_instance)   
    # print("Time taken for Gamma_m = ",time.time()-start_time) 
    # d_Gamma_m_dt = np.zeros((2*N_f,2*N_f),dtype="complex")     
    # print("Gamma_m")
    
    # Taking the real parts since all of the terms are real valued (and so any complex part should be due to numerical errors)
    d_delta_R_dt = np.real(d_delta_R_dt)     
    d_Gamma_b_dt = np.real(d_Gamma_b_dt)
    d_Gamma_m_dt = np.real(d_Gamma_m_dt)
    # time_derivative_lambda_bar = np.real(time_derivative_lambda_bar)

    # Reshaping the arrays to a single vector    
    d_delta_R_dt = np.reshape(d_delta_R_dt,2*N_b)
    d_Gamma_b_dt = np.reshape(d_Gamma_b_dt,2*N_b*2*N_b)
    d_Gamma_m_dt = np.reshape(d_Gamma_m_dt,2*N_f*2*N_f)
    # time_derivative_lambda_bar = np.reshape(time_derivative_lambda_bar,N_b*N_f)
    time_derivative_lambda_bar = np.reshape(time_derivative_lambda,N_b*int(N_f/2))
    # time_derivative_lambda_bar = np.reshape(time_derivative_lambda_bar,N_b)

    # Creating the final array to be returned 
    dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_m_dt,time_derivative_lambda_bar))
    # dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_m_dt,time_derivative_lambda))
    gc.collect()
    
    print(" Completed time =",t,".\n")

    return dydt
##############################################################################
##############################################################################
# if __name__=="__main__":
    
    # number_of_points = 10
    # positon_value_max = [10 , 10]
    # positon_value_min = [0  , 0]
    # position_space_grid = gf.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points,True)
    # print("\n Position space grid created")

    # momentum_value_max = [np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
    # momentum_value_min = [-np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,-np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
    # momentum_space_grid = gf.coordinate_array_creator_function(momentum_value_min,momentum_value_max,number_of_points,False)
    # print(" Momentum space grid created")

    # # N_b = np.size(momentum_space_grid)
    # # N_f = np.size(position_space_grid)
    # N_b = momentum_space_grid.shape[0]
    # N_f = position_space_grid.shape[0]

    # # Defining the input variables
    # J_0 = 1
    # J_0_matrix = gf.creating_J_0_matrix(position_space_grid,J_0,positon_value_max[0],spin_index=True)
    # print(" J_0 matrix created")
    # # J_0_matrix = np.diag(np.ones(N_f)*J_0,1) + np.diag(np.ones(N_f)*J_0,-1)
    # omega = 10*np.abs(J_0)*np.ones(N_b)
    # # print("omega matrix created")
    # gamma = 0.4*omega
    # # print("gamma matrix created")
    # # lmbda = np.ones(N_b)
    # lmbda = gamma/omega
    # chemical_potential_val = -5.0


    # # Volume of the system
    # volume = np.prod(np.array(positon_value_max)-np.array(positon_value_min))

    # # Defining the object which stores all the input variables
    # initial_input_variables = gf.input_variables(position_space_grid,momentum_space_grid,volume,lmbda,J_0_matrix,gamma,omega,chemical_potential_val)
    # print(" Input variables instance created")    
    
    # # Set the seed    
    # seed = 0
    # # Get the random initial matrices
    # Delta_R = gf.initialise_delta_R_matrix(N_b,seed)
    # gamma_b = gf.initialise_Gamma_b_matrix(N_b,seed)
    # gamma_m = gf.initialise_Gamma_m_matrix(N_f,seed)    
    # print("\n Initial matrices (delta_r, gamma_b and gamma_m) created")

    # lambda_bar = gf.intialising_lambda_bar(initial_input_variables)
   
    # # Initial condition to start the evolution with
    # y0 = np.concatenate((Delta_R.flatten(),gamma_b.flatten(),gamma_m.flatten(),lambda_bar.flatten()))
    # print("\n Intial numpy matrix for the evolution created.")
    # y0_lsoda = np.real(y0)

    # tracemalloc.start()
    # gc.collect()
    # profiler.snapshot()
    # profiler.display_stats()
    # profiler.compare()
    # profiler.print_trace()
    # imag_time_evo_model_solve_ivp(1.0,y0,initial_input_variables)


