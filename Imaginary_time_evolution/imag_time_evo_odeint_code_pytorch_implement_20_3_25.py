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
import torch
from Common_codes import file_with_checks_20_3_25 as fwc
from scipy.integrate import odeint
import numpy as np
from Imaginary_time_evolution import imaginary_time_evolution_functions_pytorch_implement_20_3_25 as itef
from Common_codes import class_defn_file_20_3_25 as cdf
from Common_codes import correlation_functions_file_20_3_25 as cf
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
        lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):].astype("complex")
    
    if(y.dtype == "complex"):
        delta_R = y[0:2*N_b]
        Gamma_b = y[2*N_b:((2*N_b)*(2*N_b)+2*N_b)]
        Gamma_m = y[(2*N_b)*(2*N_b)+2*N_b:((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b)]
        lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):]
    
    delta_R = torch.tensor( delta_R  ,dtype=torch.complex128)
    # Reshaping the flattened Gamma_b array to a 2N_bx2N_b
    Gamma_b = torch.tensor(np.reshape(Gamma_b,(2*N_b,2*N_b))    ,dtype=torch.complex128)
    
    # Reshaping the flattened Gamma_m array to a 2N_fx2N_f
    Gamma_m = torch.tensor(np.reshape(Gamma_m,(2*N_f,2*N_f)),dtype=torch.complex128)

    lambda_bar = torch.tensor(np.reshape(lambda_bar,(N_b,N_f)),dtype=torch.complex128) 

    # Checking the input matrices to see if they make sense
    fwc.check_bosonic_quadrature_covariance_matrix(Gamma_b)
    fwc.check_majorana_covariance_matrix(Gamma_m)   
    fwc.check_bosonic_quadrature_average_matrix(delta_R)

    # Initialising the lambda in input variable variables 
    input_variables.updating_lambda(lambda_bar)
    
    # Initialising the computational_variables class
    computed_variables_instance = cdf.computed_variables(N_b,N_f)

    # Computing the values for the computed_varaibles class
    computed_variables_instance.initialize_all_variables(input_variables,delta_R,Gamma_b)

    # Initialising the c_c correlation matrices
    correlation_matrices =cf.correlation_functions(Gamma_m,N_f)
    # start_time = time.time()
    # Equation of motion for lambda_bar
    
    time_derivative_lambda= itef.equation_of_motion_for_Non_Gaussian_parameter(delta_R,Gamma_b,Gamma_m,
                                                                                input_variables,
                                                                                computed_variables_instance,
                                                                                correlation_matrices)
    
    if(time_derivative_lambda.dtype !=torch.complex128):
        raise Exception("The time_derivative_lambda is not a complex tensor.")
    
    # Equation of motion for delta_R
    d_delta_R_dt = itef.equation_of_motion_for_bosonic_averages(delta_R,Gamma_b,Gamma_m,
                                                                time_derivative_lambda,
                                                                input_variables,
                                                                computed_variables_instance,
                                                                correlation_matrices)
    
    # Equation of motion for Gamma_b
    d_Gamma_b_dt = itef.equation_of_motion_for_bosonic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                    time_derivative_lambda,
                                                                    input_variables,
                                                                    computed_variables_instance,
                                                                    correlation_matrices)  
    
    # Equation of motion for Gamma_f
    d_Gamma_m_dt = itef.equation_of_motion_for_fermionic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                    time_derivative_lambda,
                                                                    input_variables,
                                                                    computed_variables_instance,
                                                                    correlation_matrices)   
    
    if(torch.any(torch.imag(d_delta_R_dt)>1e-10)):
        print(" WARNING: There is a large imaginary part in the d_delta_R_dt. Maximum value is:",torch.max(torch.imag(d_delta_R_dt)))
        if(torch.any(torch.imag(d_delta_R_dt)>1e-4)):
            np.save("delta_R_issue.npy",delta_R)
            raise Exception("The d_delta_R_dt has a large imaginary part.")
            
    if(torch.any(torch.imag(d_Gamma_b_dt)>1e-10)):
        print(" WARNING: There is a large imaginary part in the d_Gamma_b_dt. Maximum value is:",torch.max(torch.imag(d_Gamma_b_dt)))
    
    if(torch.any(torch.imag(d_Gamma_m_dt)>1e-10)):
        print(" WARNING: There is a large imaginary part in the d_Gamma_m_dt. Maximum value is:",torch.max(torch.imag(d_Gamma_m_dt)))
        if(torch.any(torch.imag(d_Gamma_m_dt)>1e-4)):
            np.save("Gamma_m_issue.npy",Gamma_m)
            raise Exception("The d_Gamma_m_dt has a large imaginary part.")
        
    if(torch.any(torch.imag(time_derivative_lambda)>1e-10)):
        print(" WARNING: There is a large imaginary part in the time_derivative_lambda. Maximum value is:",torch.max(torch.imag(time_derivative_lambda)))
        if(torch.any(torch.imag(time_derivative_lambda)>1e-4)):
            np.save("lambda_issue.npy",lambda_bar)
            raise Exception("The time_derivative_lambda has a large imaginary part.")
    # Taking the real parts since all of the terms are real valued (and so any complex part should be due to numerical errors)
    # d_delta_R_dt = torch.real(d_delta_R_dt)     
    # d_Gamma_b_dt = torch.real(d_Gamma_b_dt)
    # d_Gamma_m_dt = torch.real(d_Gamma_m_dt)
    
    # Remember that time derivatives are torch tensors and need to be converted to a numpy arrays
    d_delta_R_dt = np.real(np.array(d_delta_R_dt))
    d_Gamma_b_dt = np.real(np.array(d_Gamma_b_dt))
    d_Gamma_m_dt = np.real(np.array(d_Gamma_m_dt))
    time_derivative_lambda_np_array = np.real(np.array(time_derivative_lambda))

    # Reshaping the arrays to a single vector    
    d_delta_R_dt = np.reshape(d_delta_R_dt,2*N_b)
    d_Gamma_b_dt = np.reshape(d_Gamma_b_dt,2*N_b*2*N_b)
    d_Gamma_m_dt = np.reshape(d_Gamma_m_dt,2*N_f*2*N_f)
    time_derivative_lambda_np_array = np.reshape(time_derivative_lambda_np_array,N_b*N_f)

    # Creating the final array to be returned 
    dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_m_dt,time_derivative_lambda_np_array))
    # dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_m_dt,time_derivative_lambda))
    gc.collect()
    
    print(" Completed time =",t,".\n")

    return dydt
##############################################################################
##############################################################################
def imag_time_evo_model_solve_ivp_spinless_ansatz(t:np.ndarray,y:np.ndarray,input_variables:cdf.input_variables):
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
    
    delta_R = torch.tensor( delta_R  ,dtype=torch.complex128)
    # Reshaping the flattened Gamma_b array to a 2N_bx2N_b
    # Gamma_b = np.reshape(Gamma_b,(2*N_b,2*N_b))    
    Gamma_b = torch.tensor(np.reshape(Gamma_b,(2*N_b,2*N_b)) ,dtype=torch.complex128)

    # Reshaping the flattened Gamma_m array to a 2N_fx2N_f
    # Gamma_m = np.reshape(Gamma_m,(2*N_f,2*N_f))
    Gamma_m = torch.tensor(np.reshape(Gamma_m,(2*N_f,2*N_f)),dtype=torch.complex128)

    # Reshaping the spin_removed_lambda_bar array to a N_b x N_f/2 array
    lmbda_q =torch.tensor( np.reshape(lmbda_q,(N_b,int(N_f/2))),dtype=torch.complex128)  # We get N_f/2 since we have spin in this simulation.

    # input_variables.updating_lambda_bar_from_lambda(lmbda_q=lmbda_q,volume=N_b)  # Note that here we need to change the volume if we ever do. Else, it is N_b.
    input_variables.updating_lambda_from_spin_removed_lambda_bar(lmbda_q,spin_index=True,update_in_input_variables=True)  
    
    # Initialising the computational_variables class
    computed_variables_instance = cdf.computed_variables(N_b,N_f)
    # print("Completed getting computed variables instance =",t,".")

    # Computing the values for the computed_varaibles class
    computed_variables_instance.initialize_all_variables(input_variables,delta_R,Gamma_b)

    correlation_matrices =cf.correlation_functions(Gamma_m,N_f)

    # start_time = time.time()
    # Equation of motion for lambda_bar
    # time_derivative_lambda= itef.equation_of_motion_for_Non_Gaussian_parameter_lambda(delta_R,Gamma_b,Gamma_m,
    #                                                                                     input_variables,
    #                                                                                     computed_variables_instance)
    time_derivative_lambda= itef.equation_of_motion_for_Non_Gaussian_parameter_spin_modes_summed(delta_R,Gamma_b,Gamma_m,
                                                                                    input_variables,
                                                                                    computed_variables_instance,
                                                                                    correlation_matrices)
    # time_derivative_lambda = np.zeros((N_b,))
    time_derivative_lambda_bar = input_variables.updating_lambda_from_spin_removed_lambda_bar(time_derivative_lambda,
                                                                                spin_index=True ,update_in_input_variables=False)

    # Equation of motion for delta_R
    d_delta_R_dt = itef.equation_of_motion_for_bosonic_averages(delta_R,Gamma_b,Gamma_m,
                                                                time_derivative_lambda_bar,
                                                                input_variables,
                                                                computed_variables_instance,correlation_matrices)

    # Equation of motion for Gamma_b
    d_Gamma_b_dt = itef.equation_of_motion_for_bosonic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                    time_derivative_lambda_bar,
                                                                    input_variables,
                                                                    computed_variables_instance,
                                                                    correlation_matrices)  
    
    # Equation of motion for Gamma_f
    d_Gamma_m_dt = itef.equation_of_motion_for_fermionic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                    time_derivative_lambda_bar,
                                                                    input_variables,
                                                                    computed_variables_instance,
                                                                    correlation_matrices)   
    
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