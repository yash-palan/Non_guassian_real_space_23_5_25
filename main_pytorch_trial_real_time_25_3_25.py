# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:41:53 2023

@author: Yash_palan
"""
#########################################
#########################################
from Common_codes import file_with_checks_20_3_25 as fwc
import scipy.integrate as integrate
import numpy as np
# from Imaginary_time_evolution.imag_time_evo_odeint_code_pytorch_implement_20_3_25 import imag_time_evo_model_solve_ivp 
from Real_time_evolution.real_time_evo_odeint_code_trial_20_3_25 import real_time_evo_model_solve_ivp 
import time as te
import torch
# import h5py
import os
import gc
import thewalrus as tw
from Common_codes import generic_codes_20_3_25 as gcs
from Common_codes import class_defn_file_20_3_25 as cdf
import matplotlib.pyplot as plt
#########################################
#########################################

def random_gamma_m_haar_distributed(N_f):
    O_m = tw.random.random_interferometer(2*N_f, real=True) # type: ignore
    sigma = np.array(gcs.sigma(N_f)).real
    gamma_m = -np.matmul(np.matmul(O_m,sigma),O_m.T)
    return gamma_m

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

if __name__=="__main__":

    gc.collect()

    start_time = te.time()

    t_min = 0
    t_max = 20

    t_to_save = 20

    time = np.linspace(t_min, t_max, 201)         # Real/ Imaginary time used in the evolution equation

    t_span = (t_min,t_max)

    #region ############### Setting up the real space and fourier space grid of the system ###############
    number_of_points = 10

    positon_value_max = [10,10]
    positon_value_min = [0,0]

    momentum_value_max = [ np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points , np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points ]
    momentum_value_min = [ -np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points , -np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points ]
    momentum_space_grid = gcs.coordinate_array_creator_function(momentum_value_min,momentum_value_max,number_of_points,False)
    momentum_space_grid_tensor = torch.tensor(momentum_space_grid,dtype=torch.complex128)
    # Remember that the system is a 2D one (for comparison with Shi et al. paper)
    # Hence we need to create a 2D grid for the position and momentum space
    position_space_grid = gcs.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points,True)
    position_space_grid_tensor = torch.tensor(position_space_grid,dtype=torch.complex128)
    print("\n Position space grid created")

    boson_space_grid = gcs.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points,False)
    boson_space_grid_tensor = torch.tensor(boson_space_grid,dtype=torch.complex128)
    print(" Momentum space grid created")

    #endregion

    N_b = boson_space_grid.shape[0]
    N_f = position_space_grid.shape[0]

    # Volume of the system
    volume = np.prod(np.array(positon_value_max)-np.array(positon_value_min))

    #region ############### Defining the input variables of the Hamiltonian ###############

    J_0 = -1
    J_0_matrix = gcs.creating_J_0_matrix(position_space_grid,J_0,positon_value_max[0],spin_index=True)
    J_0_tensor = torch.tensor(J_0_matrix,dtype=torch.complex128)
    # np.save("J_0_matrix.npy",J_0_matrix)
    print(" J_0 matrix created")
    plt.pcolormesh(J_0_matrix)
    plt.title("J_0 matrix")
    plt.colorbar()
    plt.show()

    # J_0_matrix = np.diag(np.ones(N_f)*J_0,1) + np.diag(np.ones(N_f)*J_0,-1)
    # omega_0 = 10*np.abs(J_0)
    omega_0 = 0.2*np.abs(J_0)
    # omega = 10*np.abs(J_0)*np.identity(N_b)
    omega = omega_0*np.identity(N_b)
    omega_tensor = torch.tensor(omega,dtype=torch.complex128)
    # np.save("omega_matrix.npy",omega)
    print(" omega created")
    plt.pcolormesh(omega)
    plt.title("omega")
    plt.colorbar()
    plt.show()
    # Here, the gamma we give in is assumed to be normalised with the 1/sqrt{V} factor so that we 
    # work in the computational basis from the start
    # gamma_0 = 0.5*omega_0
    gamma_0 = 0.1*omega_0
    # gamma_0 = 2*omega_0
    gamma = gamma_0*np.append(np.identity(N_b),np.identity(N_b),axis = 1)
    gamma_tensor = torch.tensor(gamma,dtype=torch.complex128)
    print(" gamma created")
    # print(np.shape(gamma))
    plt.pcolormesh(gamma)
    plt.title("gamma")
    plt.colorbar()
    plt.show()
    # Similarly, we have to take a lambda with is divied by the 1/sqrt{V} factor from the start to prevent any 
    # normalisation issues

    # lmbda = gamma_0/omega_0*np.random.rand(N_b,N_f)
    # lmbda = np.load("imag_time_evo_final_lambda_mu_-5.1_t_95.npy")
    # lmbda_spin_removed = np.load("imag_time_evo_final_lambda_mu_-5.5_t_70.npy")


    # Initial chemical potential value
    # mu_ini = -5.5
#     mu_ini = -5.1
    mu_ini = -5.1

    # Final chemical potential value
    # chemical_potential_val = -5.0
    # chemical_potential_val = -4.5
    # chemical_potential_val = -5.1
    chemical_potential_val = -5.5


    # lmbda = np.append(lmbda_spin_removed,lmbda_spin_removed,axis=1)
    base_path = os.getcwd()

#     complete_path = base_path + '/data/Imaginary time evolution/mu='+str(mu_ini)+'/t=70'
#     complete_path = base_path + '/data/Real time evolution/'
    # complete_path = base_path +"\\data\\Real time evolution\\omega_0_0.2_J_0\\mu_-5.1_to_-5.5\\gamma_0.5_omega_0\\t=10\\"
#     filename = complete_path+"/imag_time_evo_final_lambda_mu_-5.5_t_70.npy"

    complete_path = base_path + '/data/Imaginary time evolution/mu='+str(mu_ini)+'/t=95'
    filename = complete_path+"/imag_time_evo_final_lambda_mu_-5.1_t_95.npy"
#     filename = complete_path+"real_time_evo_final_lambda_mu_ini_-5.1_mu_final_-5.5_t_10.npy"

    # filename = "real_time_evo_final_lambda_mu_ini_-5.1_mu_final_-5.5_t_10.npy"

    # lmbda = np.load("imag_time_evo_final_lambda_mu_-5.5_t_70.npy")
    lmbda = np.load(filename)
    print(lmbda.shape)
    # lmbda = np.load("real_time_evo_final_lambda_mu_ini_-5.1_mu_final_-4.0_t_40.npy")
    lmbda_tensor = torch.tensor(lmbda,dtype=torch.complex128)


    #endregion

    # Defining the object which stores all the input variables
    fourier_transform_matrix = gcs.creating_fourier_matrix(momentum_space_grid,position_space_grid)
    fourier_transform_matrix_tensor = torch.tensor(fourier_transform_matrix,dtype=torch.complex128)

    initial_input_variables = cdf.input_variables(position_space_grid_tensor,boson_space_grid_tensor,fourier_transform_matrix_tensor,
                                                    lmbda_tensor,J_0_tensor,gamma_tensor,omega_tensor,chemical_potential_val)

    print(" Input variables instance created")    


    #region ############### Get the random initial matrices ###############

    # Set the seed    
    # seed = np.random.randint(0,1000)


    # Get the random initial matrices

    # Get the random delta_R matrix
    # Delta_R = imf.initialise_delta_R_matrix(N_b,seed)
    # Delta_R = np.load("imag_time_evo_final_delta_r_mu_-5.1_t_95.npy")
    # Delta_R = np.load("imag_time_evo_final_delta_r_mu_-5.5_t_25.npy")
    # Delta_R = np.load("imag_time_evo_final_delta_r_mu_-5.5_t_70.npy")

    # Delta_R = np.load("real_time_evo_final_delta_r_mu_ini_-5.1_mu_final_-4.0_t_40.npy")
    filename = complete_path+"/imag_time_evo_final_delta_r_mu_-5.1_t_95.npy"
#     filename = complete_path+"real_time_evo_final_delta_r_mu_ini_-5.1_mu_final_-5.5_t_10.npy"
    # filename = "real_time_evo_final_delta_r_mu_ini_-5.1_mu_final_-5.5_t_80.npy"

    # filename = complete_path+"/imag_time_evo_final_delta_r_mu_-5.5_t_70.npy"    
#     filename = "real_time_evo_final_delta_r_mu_ini_-5.1_mu_final_-5.5_t_30.npy"
    Delta_R = np.load(filename)
    Delta_R_tensor = torch.tensor(Delta_R,dtype=torch.complex128)

    # np.save("initial_delta_r_mu_"+str(chemical_potential_val)+".npy", Delta_R)

    # Get the random Gamma_b matrix
    # S_b = tw.random.random_symplectic(N_b) # type: ignore
    # gamma_b = np.matmul(S_b,S_b.T)
    # gamma_b = torch.eye(2*N_b,dtype = torch.complex128)

    filename = complete_path+"/imag_time_evo_final_gamma_b_mu_-5.1_t_95.npy"
#     filename = complete_path+"real_time_evo_final_gamma_b_mu_ini_-5.1_mu_final_-5.5_t_10.npy"
#     filename = "real_time_evo_final_gamma_b_mu_ini_-5.1_mu_final_-5.5_t_30.npy"
    # filename = complete_path+"/imag_time_evo_final_gamma_b_mu_-5.5_t_70.npy"

    gamma_b = np.load(filename)
    gamma_b_tensor = torch.tensor(gamma_b,dtype=torch.complex128)



    # Get the random Gamma_m matrix
    # gamma_m = random_gamma_m_haar_distributed(N_f)
    filename = complete_path+"/imag_time_evo_final_gamma_m_mu_-5.1_t_95.npy"
#     filename = complete_path+"real_time_evo_final_gamma_m_mu_ini_-5.1_mu_final_-5.5_t_10.npy"
    # filename = "real_time_evo_final_gamma_m_mu_ini_-5.1_mu_final_-5.5_t_30.npy"

    # filename = complete_path+"/imag_time_evo_final_gamma_m_mu_-5.5_t_70.npy"
    gamma_m = np.load(filename)
    gamma_m_tensor = torch.tensor(gamma_m,dtype=torch.complex128)

    ini_phase = 0
    
    print("\n Initial matrices (delta_r, gamma_b and gamma_m) created")
    #endregion

    #region ############### Storing initial matrices to a file ###############
    # 
    # np.save("initial_delta_r_mu_"+str(chemical_potential_val)+".npy", Delta_R)
    # np.save("initial_gamma_b_mu_"+str(chemical_potential_val)+".npy", gamma_b )
    # np.save("initial_gamma_m_mu_"+str(chemical_potential_val)+".npy", gamma_m )
    # np.save("initial_lambda_matrix_mu_"+str(chemical_potential_val)+".npy", lmbda  )
    # print(" Saved the initial matrices files for access in case something goes wrong.")   

    # Check the initial matrices to have the correct properties
    fwc.check_bosonic_quadrature_covariance_matrix(gamma_b_tensor)
    fwc.check_bosonic_quadrature_average_matrix(Delta_R_tensor)
    fwc.check_majorana_covariance_matrix(gamma_m_tensor) 
    print(" Checked the initial matrices for having the correct properties")
    #endregion

    # create_hdf5_file("delta_R_time_evo"+"_mu_final_"+str(chemical_potential_val)+".h5")
    # create_hdf5_file("gamma_b_time_evo"+"_mu_final_"+str(chemical_potential_val)+".h5")
    # create_hdf5_file("gamma_m_time_evo"+"_mu_final_"+str(chemical_potential_val)+".h5")
    # create_hdf5_file("lambda_time_evo"+"_mu_final_"+str(chemical_potential_val)+".h5")

    # create_hdf5_file("d_delta_R_time_evo"+"_mu_final_"+str(chemical_potential_val)+".h5")
    # create_hdf5_file("d_gamma_m_time_evo"+"_mu_final_"+str(chemical_potential_val)+".h5")
    # create_hdf5_file("d_gamma_b_time_evo"+"_mu_final_"+str(chemical_potential_val)+".h5")
    # create_hdf5_file("time_derivative_data_time_evo"+"_mu_final_"+str(chemical_potential_val)+".h5")

    #region ############### Plotting the initial matrices to check for randomisation ###############
    # 
    plt.plot(Delta_R)
    plt.title("Initial Delta_R matrix")
    plt.show()

    plt.pcolormesh(gamma_b)
    plt.colorbar()
    plt.title("Initial gamma_b matrix")
    plt.show()

    plt.pcolormesh(gamma_m)
    plt.colorbar()
    plt.title("Initial gamma_m matrix")
    plt.show()

    plt.plot(lmbda)
    plt.title("Initial lambda matrix")
    plt.show()
    plt.xlabel("Index")
    plt.ylabel(r"$\lambda_q$")
    #endregion

    lambda_bar = lmbda

    #region ############### Initial value to start the evolution with ###############
    y0 = np.concatenate((Delta_R.flatten(),gamma_b.flatten(),gamma_m.flatten(),lambda_bar.flatten(),np.array([ini_phase]))).astype(dtype="complex")
    print("\n Intial numpy matrix for the evolution created.")
    #endregion 

    #region ###############  Selecting the model to be used for time evolution (Real/Imaginary time evolution) ###############
    # model = real_time_evo_model
    model_solve_ivp = real_time_evo_model_solve_ivp

    #region ############### Setting atolerance and rtolerance  ###############
    # rtolerance = 1e-3
    # atolerance = 1e-7
    rtolerance = 1e-4
    atolerance = 1e-8

    # rtolerance = 1e-2 
    # atolerance = 1e-6
    #endregion

    #region ############### Numerically time evolving the ODE ###############
    # Numerically time evolving the ODE
    sol_solve_ivp = integrate.solve_ivp(model_solve_ivp, t_span, y0, args=(initial_input_variables,) ,t_eval=time,method='RK45',rtol=rtolerance,atol=atolerance)

    print(" Completed solve_ivp modeule. Time taken for the same is:",te.time()-start_time)

    # Saving data to a file so it can be extracted and plotted later  
    sol = sol_solve_ivp.y.T


    #endregion


    #region ############### Saving data of the final ODE ###############

    # if(True in set(np.reshape(np.imag(sol)>1e-7,(-1) ).tolist()) ):
    #     print("\n There is a large imaginary part in the solution. Check the code and the evolution again.")
    # else:

    # Saving data to a file so it can be extracted and plotted later  
    sol = sol_solve_ivp.y.T
    Gamma_b_final = np.reshape(np.real(sol[-1,2*N_b:2*N_b +(2*N_b)*(2*N_b)]),(2*N_b,2*N_b) ) 
    delta_r_final = np.real(sol[-1,0:2*N_b])
    Gamma_m_final = np.reshape(np.real(sol[-1,2*N_b+(2*N_b)*(2*N_b):2*N_b+(2*N_b)*(2*N_b) + (2*N_f)*(2*N_f)] ),(2*N_f,2*N_f))

    location = base_path + '/data/Real time evolution/'

    np.save(location+'time_data.npy', sol_solve_ivp.t)

    #  Saving to numpy files (for speed, hopefully? )
    np.save(location+"real_time_evo_final_delta_r_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy",
            np.real(sol[-1,0:2*N_b]))
    np.save(location+"real_time_evo_final_gamma_b_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy",
            np.reshape( np.real(sol[-1,2*N_b:2*N_b +(2*N_b)*(2*N_b)]),(2*N_b,2*N_b) )  )
    np.save(location+"real_time_evo_final_gamma_m_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy",
            np.reshape( np.real(sol[-1,2*N_b+(2*N_b)*(2*N_b):2*N_b+(2*N_b)*(2*N_b) + (2*N_f)*(2*N_f)] ),(2*N_f,2*N_f) )   )
    # np.save('imag_time_evo_final_lambda_bar.npy', np.reshape(np.real(sol[-1,2*N_b+(2*N_b)*(2*N_b) + (2*N_f)*(2*N_f):] ),(N_b,N_f) )  )
    np.save(location+"real_time_evo_final_lambda_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy",
            np.reshape(np.real(sol[-1,2*N_b+(2*N_b)*(2*N_b) + (2*N_f)*(2*N_f):-1] ),(N_b,N_f) )  )
    np.save(location+"real_time_evo_final_phase_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy",
            sol[-1] )

    Gamma_b_complete = np.real(sol[:,2*N_b:2*N_b +(2*N_b)*(2*N_b)]) 
    delta_r_complete = np.real(sol[:,0:2*N_b])
    Gamma_m_complete = np.real(sol[:,2*N_b+(2*N_b)*(2*N_b):2*N_b+(2*N_b)*(2*N_b) + (2*N_f)*(2*N_f)] )


    np.save(location+"real_time_evo_delta_r_complete_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy",
            delta_r_complete)
    np.save(location+"real_time_evo_gamma_b_complete_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy", 
            Gamma_b_complete )
    np.save(location+"real_time_evo_gamma_m_complete_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy", 
            Gamma_m_complete )
    np.save(location+"real_time_evo_lambda_complete_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy", 
            np.real(sol[:,2*N_b+(2*N_b)*(2*N_b) + (2*N_f)*(2*N_f):-1] ) )
    np.save(location+"real_time_evo_phase_complete_mu_ini_"+str(mu_ini)+"_mu_final_"+str(chemical_potential_val)+"_t_"+str(t_to_save)+".npy", 
            sol[:,-1])

    print(" Saved all the data to the approproate files.")

    # Plotting final matrices

    # Plotting gamma_b matrix
    plt.pcolormesh(Gamma_b_final)
    plt.colorbar()
    plt.title("Final gamma_b matrix")
    plt.show()

    # Plotting delta_r matrix
    plt.plot(delta_r_final)
    plt.title("Final delta_r matrix")
    plt.show()

    # Plotting gamma_m matrix
    plt.pcolormesh(Gamma_m_final)
    plt.colorbar()
    plt.title("Final gamma_m matrix")
    plt.show()
    #endregion
    print("\n Finally done")