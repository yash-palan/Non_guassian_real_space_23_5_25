# -*- coding: utf-8 -*-
"""
Created on Oct 2 2024

@author: Yash_palan

This contains code to extract all the physical observables
"""
import matplotlib.pyplot as plt
import numpy as np
from Common_codes import correlation_functions_file_13_9_24 as cff
from Common_codes import evolution_video_creator as evc
from Common_codes import generic_codes_13_9_24 as gc    
from Common_codes import class_defn_file_13_9_24 as cdf

#################################################
#################################################
# Function for converting data from fourier space to real space using 
# the Fast Fourier transform in python 

def fourier_transform_delta_r(delta_r:np.ndarray,position_array:np.ndarray,momentum_array:np.ndarray,volume:float,spin_index):
    """
    This function just does the Fourier transform of the delta_r array (since we treat bosons in the Fourier space, while
    we treat the Fermions in the position space). Also, the relationship 

    """
    delta_x = delta_r[0:int(delta_r.shape[0]/2)]
    delta_p = delta_r[int(delta_r.shape[0]/2):]

    # if(spin_index==True):
    #     position_array_new = position_array[0:int(position_array.shape[0]/2),:]
    # else:
    #     position_array_new = position_array

    position_array_new = position_array
    expectation_value_of_b_k = 1/2*(delta_x+1j*delta_p) 
    if(delta_x.shape[0]!=delta_p.shape[0]):
        raise Exception("The size of the delta_x and delta_p arrays are not the same. Check the breakup again.")
    
    if(position_array.ndim==1):
        matrix = np.einsum('i,k->ki',position_array_new,momentum_array)
    if(position_array.ndim>1):
        matrix = np.einsum('id,kd->ki',position_array_new,momentum_array)
    
    expectation_value_of_b_i =(1/np.sqrt(volume))*np.einsum('k,ki->i',expectation_value_of_b_k,np.exp(1j*matrix))
    
    delta_x_position_space = 2*np.real(expectation_value_of_b_i)
    delta_p_position_space = 1j*(np.conjugate( expectation_value_of_b_i) - expectation_value_of_b_i)

    return(delta_x_position_space,delta_p_position_space)

#################################################
#################################################
# Functions for plotting 2D systems

def function_for_plotting_delta_r_in_real_space(delta_r:np.ndarray,position_array:np.ndarray,momentum_array:np.ndarray,volume,spin_index=True):
    if(spin_index==True):
        position_array_new = position_array[0:int(position_array.shape[0]/2),:]
    else:
        position_array_new = position_array
    # position _array extraction    
    
    N_f = position_array.shape[0]/2
    dim_position = position_array.shape[1]
    size_of_one_dimension_momentum_array = int(np.power(N_f,1/dim_position))

    # Since we are working with a 2D system, we can directly repeat the values 2 times
    # momentum_dimensions_extraction = np.reshape(np.transpose(momentum_array),
    #                                         (dim_position,size_of_one_dimension_momentum_array,size_of_one_dimension_momentum_array))
    # X = momentum_dimensions_extraction[0]
    # Y = momentum_dimensions_extraction[1]

    position_dimensions_extraction = np.reshape(np.transpose(position_array_new),
                                            (dim_position,size_of_one_dimension_momentum_array,size_of_one_dimension_momentum_array))
    X = position_dimensions_extraction[0]
    Y = position_dimensions_extraction[1]
    
    delta_x_position_space,delta_p_position_space = fourier_transform_delta_r(delta_r,position_array_new,momentum_array,volume,spin_index)
    
    plt.pcolormesh(X,Y,np.reshape(delta_x_position_space,(size_of_one_dimension_momentum_array,size_of_one_dimension_momentum_array)))
    plt.title('Delta_r in real space')
    plt.xlabel("Lattice sites")
    plt.ylabel("Lattice sites")
    # plt.clim(0,1)
    plt.colorbar()

    plt.show()

##################################################
##################################################
# Fermionic order parameters
##################################################
##################################################

def extraction_of_c_dagger_c_from_Gamma_m_for_all_time(Gamma_m_data,N_f):
    time_length = Gamma_m_data.shape[0]
    final_array = np.zeros((time_length,N_f*N_f))
    count = 0
    for ele in Gamma_m_data:
        final_array[count] = np.reshape(  cff.c_dagger_c_expectation_value_matrix_creation(np.reshape(ele,(2*N_f,2*N_f)),N_f),-1 )
        count += 1
    return(final_array)

def extraction_of_c_c_from_Gamma_m_for_all_time(Gamma_m_data,N_f):
    time_length = Gamma_m_data.shape[0]
    final_array = np.zeros((time_length,N_f*N_f))
    count = 0
    for ele in Gamma_m_data:
        final_array[count] = np.reshape(  cff.c_c_expectation_value_matrix_creation(np.reshape(ele,(2*N_f,2*N_f)),N_f),-1 )
        count += 1
    return(final_array)

def extraction_of_supeconducting_order_parameter_real_space(c_c_data,N_f):
    """
    c_c_data:   Size: N_f x N_f
                Description: The data for the c_c expectation value
    N_f:        Size: int
                Description: The number of fermionic sites
    
    Returns:    Size: N_f/2 x N_f/2
                Description: <c_{i,up}c_{j,down}>
    """
    
    N_f_spinless = int(N_f/2)
    return(c_c_data[0:N_f_spinless,N_f_spinless:])

def extraction_of_supeconducting_order_parameter(c_c_data,position_array,momentum_array,volume):
    """
    c_c_data:   Size: N_f x N_f
                Description: The data for the c_c expectation value
    N_f:        Size: int
                Description: The number of fermionic sites
    
    Returns:    Size: N_f/2 x N_f/2
                Description: <c_{i,up}c_{j,down}>
    """
    # Computing the matrix for \va{r}_{i}.\va{k}, for 1D, and higher dimensional systems
    N_f = np.shape(position_array)[0]
    N_f_spinless = int(N_f/2)

    sc_order_parameter_real_space = c_c_data[0:N_f_spinless,N_f_spinless:]

    if(len(np.shape(position_array))==1):
        # For 1D systems
        # The order in which we take the position array does not matter since we have the same order for both the spins
        matrix = np.einsum('i,q->iq',position_array[0:N_f_spinless],momentum_array)
    
    if(len(np.shape(position_array))!=1):
        # For higher dimensional systems (this is r_i ad then k)
        # The order in which we take the position array does not matter since we have the same order for both the spins
        matrix = np.einsum('ik,qk->iq',position_array[0:N_f_spinless],momentum_array)
        
        # Check the sign for the exponential here, that is important
    final_matrix = (1/volume)*np.einsum('ij,ik,lj->kl',sc_order_parameter_real_space,np.exp(1j*matrix),np.exp(-1j*matrix))
         
    return(final_matrix)

def extraction_of_spin_number_density(c_dagger_c_data,N_f):
    """
    c_dagger_c_data:   Size: N_f x N_f
                        Description: The data for the c_dagger_c expectation value
    N_f:                Size: int
                        Description: The number of fermionic sites
    
    Returns:            Size: [up_spin_density,down_spin_density]
                        Description: [sum_i <c_{i,up}c_{i,up}>,sum_i <c_{i,down}c_{i,down>}]
    """
    
    N_f_spinless = int(N_f/2)
    diagonal_part = np.diag(c_dagger_c_data)
    up_spin_density = np.sum(diagonal_part[0:N_f_spinless])/N_f
    down_spin_density = np.sum(diagonal_part[0:N_f_spinless])/N_f
    return(up_spin_density,down_spin_density)

##################################################
##################################################
# Creating evolution videos for the fermionic order parameters
##################################################
##################################################

def superconducting_order_parameter_video_and_plot_function(Gamma_m_data,position_array,momentum_array,volume,N_f)->None:

    c_c_data = extraction_of_c_c_from_Gamma_m_for_all_time(Gamma_m_data,N_f)

    superconducting_order_parameter = extraction_of_supeconducting_order_parameter(c_c_data,position_array,momentum_array,volume)
    filename = "c_c_expectation_momentum_space_evolution"
    
    N_f_spinless = int(N_f/2)

    # Creating the video
    evc.function_to_create_2D_video(data=superconducting_order_parameter,square_size=N_f_spinless,filename=filename)
    
    # Creating the plot for the final time step
    plt.pcolormesh(np.reshape(superconducting_order_parameter[-1,:],(N_f_spinless,N_f_spinless)))
    plt.colorbar()
    plt.title("c_c expectation value in real space")
    plt.xlabel(r"$\vec{r},\uparrow$")
    plt.ylabel(r"$\vec{r},\downarrow")

    return

def c_dagger_c_video_and_plot_function(Gamma_m_data,N_f)->None:

    c_dagger_c_data = extraction_of_c_dagger_c_from_Gamma_m_for_all_time(Gamma_m_data,N_f)

    filename = "c_c_expectation_real_space_evolution"
 
    # Creating the video
    evc.function_to_create_2D_video(data=c_dagger_c_data,square_size=N_f,filename=filename)
    
    # Creating the plot for the final time step
    plt.pcolormesh(np.reshape(c_dagger_c_data[-1,:],(N_f,N_f)))
    plt.colorbar()
    plt.title("c_dagger_c expectation value in real space")
    plt.xlabel("Lattice sites")
    plt.ylabel("Lattice sites")

    return

##################################################
##################################################
if __name__ == "__main__":
    # Loading the data
    filename = "imag_time_evo_gamma_m_complete.npy"
    Gamma_m_data = np.load(filename)
    N_f = 100

    #######################################
    number_of_points = 10
    positon_value_max = [10 , 10]
    positon_value_min = [0  , 0]
    position_space_grid = gc.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points,True)
    N_f = np.shape(position_space_grid)[0]
    print("Position space grid created")

    momentum_value_max = [np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
    momentum_value_min = [-np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,-np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
    momentum_space_grid = gc.coordinate_array_creator_function(momentum_value_min,momentum_value_max,number_of_points,False)
    N_b = np.shape(momentum_space_grid)[0]
    print("Momentum space grid created")

    volume = np.prod(np.array(positon_value_max)-np.array(positon_value_min))
    
    #######################################
    # Plotting the superconducting order parameter
    superconducting_order_parameter_video_and_plot_function(Gamma_m_data,position_space_grid,momentum_space_grid,volume,N_f)
    c_dagger_c_video_and_plot_function(Gamma_m_data,N_f)
    plt.show()
    #######################################

    lambda_bar_data = np.load("imag_time_evo_lambda_bar.npy")

    input_variables_new = cdf.input_variables(position_array=position_space_grid,momentum_array=momentum_space_grid,lmbda=np.array([0,0]),J_0=np.array([0,0]),gamma=np.array([0,0]),omega=np.array([0,0]))
    input_variables_new.update_lmbda_from_lmbda_bar(lambda_bar=np.reshape(lambda_bar_data[-1],(2*N_b,N_f)))
    print(input_variables_new.lmbda)
    
    #######################################
    delta_r_data = np.load("imag_time_evo_delta_r.npy")
    function_for_plotting_delta_r_in_real_space(delta_r = delta_r_data[-1],position_array=position_space_grid,momentum_array=momentum_space_grid,volume=volume,spin_index=True)
 
    
 