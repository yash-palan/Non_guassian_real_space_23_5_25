# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 06:40:44 2023

@author: Yash_palan

This code contains the code that will

"""
#########################################
#########################################
from ast import Del
import gc
import h5py
import torch
import numpy as np
from Real_time_evolution import real_time_evolution_functions_20_3_25 as rtef
from Common_codes import class_defn_file_20_3_25 as cdf
from Common_codes import correlation_functions_file_20_3_25 as cf
from Common_codes import hamiltonian_derivative_matrices_20_3_25 as hdm
from Common_codes import file_with_checks_20_3_25 as fwc
# import csv
import pandas as pd
##############################################################################
##############################################################################
# def create_hdf5_file(filename):
#     with h5py.File(filename, "w") as f:
#         # Create a dataset with maxshape=(None,) so it can grow
#         f.create_dataset("time_series_real", shape=(0,0), maxshape=(None,None), dtype="f8")
#         f.create_dataset("time_series_imag", shape=(0,0), maxshape=(None,None), dtype="f8")
#         print("HDF5 file initialized.")

# # Function to append data to the existing dataset
# def append_to_hdf5(new_data,filename):
#     with h5py.File(filename, "a") as f:
#         # Note that the first dimension is the value of parameter, i.e. dset[:,1] is the value of the data at time t=1

#         dset_real = f["time_series_real"]
#         dset_imag = f["time_series_imag"]
#         old_size = dset_real.shape[1]  # Get the current size
#         if(dset_real.shape[1] == 0):
#             dset_real.resize((len(new_data),1))
#             dset_imag.resize((len(new_data),1))

#             dset_real[:,0] = new_data.real  # Append new data
#             dset_imag[:,0] = new_data.imag  # Append new data
#             # print(f"Appended {len(new_data)} entries to the dataset.") 
#         else:
#             second_size = dset_real.shape[0]  # Get the current size
#             new_size = old_size + 1  # Calculate new size
#             dset_real.resize((second_size,new_size))  # Resize the dataset
#             dset_imag.resize((second_size,new_size))  # Resize the dataset
#             dset_real[:,new_size-1] = new_data.real  # Append new data
#             dset_imag[:,new_size-1] = new_data.imag  # Append new data
#             # print(f"Appended {len(new_data)} entries to the dataset.")
#     return

def write_to_excel(filename,data):
    df = pd.DataFrame(data)
    df.to_excel(filename,index=False, header=False)
    return

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
        lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):-1].astype("complex")
        phase_val = y[-1].astype("complex")
    
    if(y.dtype == "complex"):
        delta_R = y[0:2*N_b]
        Gamma_b = y[2*N_b:((2*N_b)*(2*N_b)+2*N_b)]
        Gamma_m = y[(2*N_b)*(2*N_b)+2*N_b:((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b)]
        # lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):]
        # lmbda_q = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):]
        lambda_bar = y[((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b):-1]
        phase_val = y[-1]

    # append_to_hdf5(delta_R,"delta_R_time_evo"+"_mu_final_"+str(input_variables.chemical_potential_val)+".h5")
    # append_to_hdf5(Gamma_b,"gamma_b_time_evo"+"_mu_final_"+str(input_variables.chemical_potential_val)+".h5")
    # append_to_hdf5(Gamma_m,"gamma_m_time_evo"+"_mu_final_"+str(input_variables.chemical_potential_val)+".h5")
    # append_to_hdf5(lambda_bar,"lambda_time_evo"+"_mu_final_"+str(input_variables.chemical_potential_val)+".h5")
    
    delta_R = torch.tensor( delta_R  ,dtype=torch.complex128)
    # Reshaping the flattened Gamma_b array to a 2N_bx2N_b
    Gamma_b = torch.tensor(np.reshape(Gamma_b,(2*N_b,2*N_b))    ,dtype=torch.complex128)
    
    # Reshaping the flattened Gamma_m array to a 2N_fx2N_f
    Gamma_m = torch.tensor(np.reshape(Gamma_m,(2*N_f,2*N_f)),dtype=torch.complex128)

    lambda_bar = torch.tensor(np.reshape(lambda_bar,(N_b,N_f)),dtype=torch.complex128) 

    # Storting so that I can observe how things are changing
    # write_to_excel("delta_r.xlsx",delta_R)
    # write_to_excel("gamma_m.xlsx",Gamma_m)
    # write_to_excel("gamma_b.xlsx",Gamma_b)
    # write_to_excel("lambda_bar.xlsx",lambda_bar)

    # Initialising the lambda in input variable variables 
    input_variables.updating_lambda(lambda_bar)
    
    # Initialising the computational_variables class
    computed_variables_instance = cdf.computed_variables(N_b,N_f)
    
    
    # Checking if the arrays the required properties
    fwc.check_bosonic_quadrature_average_matrix(delta_R)
    fwc.check_bosonic_quadrature_average_matrix(Gamma_b)
    fwc.check_majorana_covariance_matrix(Gamma_m)

    # Computing the values for the computed_varaibles class
    computed_variables_instance.initialize_all_variables(input_variables,delta_R,Gamma_b)
    if(computed_variables_instance.J_i_j_nan_inf_flag):
        np.save("delta_R_issue.npy",delta_R)
        np.save("Gamma_b_issue.npy",Gamma_b)
        np.save("Gamma_m_issue.npy",Gamma_m)
        np.save("lambda_issue.npy",lambda_bar)
        raise Exception("The J_i_j is inf or Nan. Saved data for checking.")

    # Initialising the c_c correlation matrices
    correlation_matrices =cf.correlation_functions(Gamma_m,N_f)
    # start_time = time.time()
    # Equation of motion for lambda_bar
    print(" chemical potential:",input_variables.chemical_potential_val)
    # start_time = time.time()
    # Equation of motion for lambda_bar
    # Equation of motion for phase_val
    phase_time_derivative = hdm.energy_expectation_value(delta_R,Gamma_b,Gamma_m,
                            input_variables,
                            computed_variables_instance,
                            correlation_matrices)
    print(" Energy expectation value:",phase_time_derivative)
    time_derivative_lambda= rtef.equation_of_motion_for_Non_Gaussian_parameter(delta_R,Gamma_b,Gamma_m,
                                                                                input_variables,
                                                                                computed_variables_instance,
                                                                                correlation_matrices)
    if(time_derivative_lambda.dtype !=torch.complex128):
        raise Exception("The time_derivative_lambda is not a complex tensor.")
    
    # Equation of motion for delta_R
    d_delta_R_dt, phase_contribution_1 = rtef.equation_of_motion_for_bosonic_averages(delta_R,Gamma_b,Gamma_m,
                                                                time_derivative_lambda,
                                                                input_variables,
                                                                computed_variables_instance,
                                                                correlation_matrices)

    phase_time_derivative += phase_contribution_1
    del phase_contribution_1

    # Equation of motion for Gamma_b
    d_Gamma_b_dt,phase_contribution_2 = rtef.equation_of_motion_for_bosonic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                time_derivative_lambda,
                                                                input_variables,
                                                                computed_variables_instance,
                                                                correlation_matrices)  
    phase_time_derivative += phase_contribution_2

    del phase_contribution_2
    
    # Equation of motion for Gamma_f
    d_Gamma_m_dt,phase_contribution_3 = rtef.equation_of_motion_for_fermionic_covariance(delta_R,Gamma_b,Gamma_m,
                                                                time_derivative_lambda,
                                                                input_variables,
                                                                computed_variables_instance,
                                                                correlation_matrices)   
    phase_time_derivative += phase_contribution_3

    del phase_contribution_3
    
    # append_to_hdf5(d_delta_R_dt,"d_delta_R_time_evo"+"_mu_final_"+str(input_variables.chemical_potential_val)+".h5")
    # append_to_hdf5(d_Gamma_b_dt.flatten(),"d_gamma_m_time_evo"+"_mu_final_"+str(input_variables.chemical_potential_val)+".h5")
    # append_to_hdf5(d_Gamma_m_dt.flatten(),"d_gamma_b_time_evo"+"_mu_final_"+str(input_variables.chemical_potential_val)+".h5")
    # append_to_hdf5(time_derivative_lambda.flatten(),"time_derivative_data_time_evo"+"_mu_final_"+str(input_variables.chemical_potential_val)+".h5")
    # write_to_excel("d_delta_R_dt.xlsx",d_delta_R_dt)
    # write_to_excel("d_Gamma_m_dt.xlsx",d_Gamma_m_dt)
    # write_to_excel("d_Gamma_b_dt.xlsx",d_Gamma_b_dt)
    # write_to_excel("time_derivative_lambda.xlsx",time_derivative_lambda)
    
    if(torch.any(torch.imag(d_delta_R_dt)>1e-10)):
        print(" WARNING: There is a large imaginary part in the d_delta_R_dt. Maximum value is:",torch.max(torch.imag(d_delta_R_dt)))
        if(torch.any(torch.imag(d_delta_R_dt)>1e-4)):
            np.save("delta_R_issue.npy",delta_R)
            np.save("Gamma_b_issue.npy",Gamma_b)
            np.save("Gamma_m_issue.npy",Gamma_m)
            np.save("lambda_issue.npy",lambda_bar)
            raise Exception("The d_delta_R_dt has a large imaginary part.")
            
    if(torch.any(torch.imag(d_Gamma_b_dt)>1e-10)):
        print(" WARNING: There is a large imaginary part in the d_Gamma_b_dt. Maximum value is:",torch.max(torch.imag(d_Gamma_b_dt)))
    
    if(torch.any(torch.imag(d_Gamma_m_dt)>1e-10)):
        print(" WARNING: There is a large imaginary part in the d_Gamma_m_dt. Maximum value is:",torch.max(torch.imag(d_Gamma_m_dt)))
        if(torch.any(torch.imag(d_Gamma_m_dt)>1e-4)):
            np.save("delta_R_issue.npy",delta_R)
            np.save("Gamma_b_issue.npy",Gamma_b)
            np.save("Gamma_m_issue.npy",Gamma_m)
            np.save("lambda_issue.npy",lambda_bar)
            raise Exception("The d_delta_R_dt has a large imaginary part.")
        
    if(torch.any(torch.imag(time_derivative_lambda)>1e-10)):
        print(" WARNING: There is a large imaginary part in the time_derivative_lambda. Maximum value is:",torch.max(torch.imag(time_derivative_lambda)))
        if(torch.any(torch.imag(time_derivative_lambda)>1e-4)):
            np.save("delta_R_issue.npy",delta_R)
            np.save("Gamma_b_issue.npy",Gamma_b)
            np.save("Gamma_m_issue.npy",Gamma_m)
            np.save("lambda_issue.npy",lambda_bar)
            raise Exception("The d_delta_R_dt has a large imaginary part.")

    # Taking the real parts since all of the terms are real valued (and so any complex part should be due to numerical errors)
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
    # dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_m_dt,time_derivative_lambda_bar))
    dydt = np.concatenate((d_delta_R_dt,d_Gamma_b_dt,d_Gamma_m_dt,time_derivative_lambda_np_array,np.array([phase_time_derivative])))
    gc.collect()
    
    print(" Completed time =",t,".\n")

    return dydt