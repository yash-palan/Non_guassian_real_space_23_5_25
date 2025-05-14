# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 06:40:44 2023

@author: Yash_palan

This code contains the code that will

"""

import gc
import time
from scipy.integrate import odeint
import numpy as np
from Common_codes import hamiltonian_derivative_matrices_18_10_24 as hdm
from Real_time_evolution import real_time_evolution_functions_v2_19_5_24 as rtef
from Common_codes import class_defn_file_18_10_24 as cdf

def real_time_evo_model_solve_ivp(t,y:np.ndarray,input_variables:cdf.input_variables)->np.ndarray:
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
        # lmbda_q = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):].astype("complex")
        lmbda_q = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):-1].astype("complex")
        phase_val = y[-1].astype("complex")
    
    if(y.dtype == "complex"):
        delta_R = y[0:2*N_b]
        Gamma_b = y[2*N_b:((2*N_b)*(2*N_b)+2*N_b)]
        Gamma_m = y[(2*N_b)*(2*N_b)+2*N_b:((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b)]
        # lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):]
        # lmbda_q = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):]
        lmbda_q = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):-1]
        phase_val = y[-1]
    
    # Reshaping the flattened Gamma_b array to a 2N_bx2N_b
    Gamma_b = np.reshape(Gamma_b,(2*N_b,2*N_b))    
    
    # Reshaping the flattened Gamma_m array to a 2N_fx2N_f
    Gamma_m = np.reshape(Gamma_m,(2*N_f,2*N_f))
    
    # Reshaping the flattened lambda_bar array to a 2N_bxN_f array
    lmbda_q = np.reshape(lmbda_q,(N_b,))

    input_variables.updating_lambda_bar_from_lambda(lmbda_q=lmbda_q,volume=N_b)  # Note that here we need to change the volume if we ever do. Else, it is N_b.
    
    # lambda_bar = np.reshape(lambda_bar,(2*N_b,N_f))

    # lambda_bar = np.reshape(lambda_bar,(N_b,N_f)) 
    
    # Initialising the computational_variables class
    computed_variables_instance = cdf.computed_variables(N_b,N_f)

    # Computing the values for the computed_varaibles class
    computed_variables_instance.initialize_all_variables(input_variables,delta_R,Gamma_b)

    # Equation of motion for phase_val
    phase_time_derivative = hdm.energy_expectation_value(delta_R,Gamma_b,Gamma_m,input_variables,computed_variables_instance)

    # Equation of motion for lambda_bar
    time_derivative_lambda = rtef.equation_of_motion_for_Non_Gaussian_parameter_lambda(delta_R,Gamma_b,Gamma_m,
                                                                                        input_variables,
                                                                                        computed_variables_instance)

    # time_derivative_lambda = np.zeros((N_b,))
    time_derivative_lambda_bar = input_variables.updating_lambda_bar_from_lambda(lmbda_q=time_derivative_lambda,volume=N_b,
                                                                                spin_index=True ,update_in_input_variables=False)
    
    # Equation of motion for delta_R
    d_delta_R_dt, phase_contribution_1 = rtef.equation_of_motion_for_bosonic_averages(delta_R,Gamma_b,Gamma_m,
                                                                time_derivative_lambda_bar,
                                                                input_variables,
                                                                computed_variables_instance)
    phase_time_derivative += phase_contribution_1
    del phase_contribution_1

    # Equation of motion for Gamma_b
    d_Gamma_b_dt,phase_contribution_2 = rtef.equation_of_motion_for_bosonic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                    time_derivative_lambda_bar,
                                                                    input_variables,
                                                                    computed_variables_instance)  
    phase_time_derivative += phase_contribution_2
    del phase_contribution_2
    
    # Equation of motion for Gamma_f
    d_Gamma_m_dt,phase_contribution_3 = rtef.equation_of_motion_for_fermionic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                    time_derivative_lambda_bar,
                                                                    input_variables,
                                                                    computed_variables_instance)   
    phase_time_derivative += phase_contribution_3
    del phase_contribution_3
    # print("Time taken for Gamma_m = ",time.time()-start_time) 
    # d_Gamma_m_dt = np.zeros((2*N_f,2*N_f),dtype="complex")     

    
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
    # time_derivative_lambda_bar = np.reshape(time_derivative_lambda_bar,N_b)

    # Creating the final array to be returned 
    # dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_m_dt,time_derivative_lambda_bar))
    dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_m_dt,time_derivative_lambda,phase_time_derivative))
    gc.collect()
    
    print(" Completed time =",t,".\n")

    return dydt