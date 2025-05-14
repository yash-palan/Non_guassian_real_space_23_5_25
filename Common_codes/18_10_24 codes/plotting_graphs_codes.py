# -*- coding: utf-8 -*-
"""
Created on Mon July 22 10:34 2024
@author: Yash_palan
"""
import matplotlib.pyplot as plt
import numpy as np
import csv
from Common_codes import global_functions_v3_pytorch_implement_24_7_24 as gf

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

# def fourier_transform_lambda_k(lambda_k:np.ndarray,position_array:np.ndarray,momentum_array:np.ndarray,volume:float,spin_index)->np.ndarray:

#################################################
#################################################
# Functions for plotting 1D systems


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


def function_for_plotting_c_dagger_c(Gamma_m:np.ndarray,N_f):
    c_dagger_c_data = gf.c_dagger_c_expectation_value_matrix_creation(Gamma_m,N_f)
    plt.pcolormesh(np.real(c_dagger_c_data))
    plt.title("c_dagger_c expectation value")
    plt.colorbar()
    plt.show()

def function_for_plotting_c_c_dagger(Gamma_m:np.ndarray,N_f):
    c_dagger_c_data = gf.c_c_dagger_expectation_value_matrix_creation(Gamma_m,N_f)
    plt.pcolormesh(np.real(c_dagger_c_data))
    plt.title("c_c_dagger expectation value")
    plt.colorbar()
    plt.show()
    
def function_for_plotting_c_c(Gamma_m:np.ndarray,N_f):
    c_dagger_c_data = gf.c_c_expectation_value_matrix_creation(Gamma_m,N_f)
    plt.pcolormesh(np.real(c_dagger_c_data))
    plt.title("c_c expectation value")
    plt.colorbar()
    plt.show()

def function_for_plotting_c_dagger_c_dagger(Gamma_m:np.ndarray,N_f):
    c_dagger_c_data = gf.c_dagger_c_dagger_expectation_value_matrix_creation(Gamma_m,N_f)
    plt.pcolormesh(np.real(c_dagger_c_data))
    plt.title("c_dagger_c_dagger expectation value")
    plt.colorbar()
    plt.show()

def function_for_plotting_Gamma_b(Gamma_b:np.ndarray,N_b):
    plt.pcolormesh(Gamma_b)
    plt.colorbar()
    plt.title("Gamma_b plot")
    plt.show()

#################################################
#################################################
#  Function for extacting the data from the csv file

def extract_data_from_csv_file(file_name:str):
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
        for ele in data:
            if ele == []:
                data.remove(ele)
        data = np.array(data).astype("float")
    return data


def extract_data_from_csv_file_2D(file_name:str):
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
        final_data = []
        for ele in data:
            if ele == []:
                data.remove(ele)
            final_data.append(list(ele) )
        # data = np.array(data).astype("float")
        final_data = np.array(final_data)
    return data

# def plotting_graphs(y_data, t,mode):
#     delta_R = y_data[0:2*N_b]
#     Gamma_b = y_data[2*N_b:((2*N_b)*(2*N_b)+2*N_b)]
#     Gamma_f = y_data[(2*N_b)*(2*N_b)+2*N_b:((2*N_f)*(2*N_f)+2*N_b*2*N_b+2*N_b)]
    
#     if(mode==1):
#         #  Need to write the code to plot the correct thing
#         # For the Bosonic average values
#         print("mode:1")

#     elif(mode==2):
#         #  Need to write the code to plot the correct thing
#         # For the Bosonic correlation matrix values
#         print("mode:2")
        
#     elif(mode==3):
#         #  Need to write the code to plot the correct thing
#         # For the Fermionic correlation matrix values
#         print("mode:3")


#################################################
#################################################
if __name__=="__main__":
    delta_r_data = extract_data_from_csv_file("imag_time_evo_delta_r.csv")
    Gamma_m_data = extract_data_from_csv_file_2D("imag_time_evo_gamma_m.csv")
    Gamma_b_data = extract_data_from_csv_file_2D("imag_time_evo_gamma_b.csv")
    lambda_bar_data = extract_data_from_csv_file_2D("imag_time_evo_lambda_bar.csv")
    # Remember that this needs to be changed in some way so that all of this data can be taken from data files

    # N_b = 100
    # N_f = 200
    # plt.plot(np.real(np.reshape(delta_r_data,(2*N_b))))
    # plt.title("Delta_r plot")
    # plt.show()
    
    # plt.pcolormesh(np.real(np.reshape(Gamma_m_data,(2*N_f,2*N_f))))
    # plt.title("Gamma_m plot")
    # plt.colorbar()
    # plt.show()

    # plt.pcolormesh(np.real(np.reshape(Gamma_b_data,(2*N_b,2*N_b))))
    # plt.title("Gamma_b plot")
    # plt.colorbar()
    # plt.show()
    
    # plt.pcolormesh(np.real(gf.c_c_dagger_expectation_value_matrix_creation(np.reshape(Gamma_m_data,(2*N_f,2*N_f)),N_f)))
    # plt.title("c_c_dagger expectation value")
    # plt.colorbar()
    # plt.show()

    # plt.pcolormesh(np.real(gf.c_dagger_c_expectation_value_matrix_creation(np.reshape(Gamma_m_data,(2*N_f,2*N_f)),N_f)))
    # plt.title("c_dagger_c expectation value")
    # plt.colorbar()
    # plt.show()

    number_of_points = 10
    positon_value_max = [10 , 10]
    positon_value_min = [0  , 0]
    position_space_grid = gf.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points,True)
    N_f = np.shape(position_space_grid)[0]
    print("Position space grid created")

    momentum_value_max = [np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
    momentum_value_min = [-np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,-np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
    momentum_space_grid = gf.coordinate_array_creator_function(momentum_value_min,momentum_value_max,number_of_points,False)
    N_b = np.shape(momentum_space_grid)[0]
    print("Momentum space grid created")

    volume = np.prod(np.array(positon_value_max)-np.array(positon_value_min))
    
    input_variables_new = gf.input_variables(position_array=position_space_grid,momentum_array=momentum_space_grid,lmbda=np.array([0,0]),J_0=np.array([0,0]),gamma=np.array([0,0]),omega=np.array([0,0]))
    input_variables_new.update_lmbda_from_lmbda_bar(lambda_bar=np.reshape(lambda_bar_data[-1],(2*N_b,N_f)))
    print(input_variables_new.lmbda)

    # plt.pcolormesh(np.real(np.reshape(Gamma_b_data,(2*N_b,2*N_b))))
    # plt.colorbar()
    # plt.show()

    # plt.pcolormesh(np.real(  np.reshape(  Gamma_m_data, (2*N_f,2*N_f)  )  ))
    # plt.colorbar()
    # plt.show()

    delta_r_data = extract_data_from_csv_file("imag_time_evo_delta_r.csv")
    function_for_plotting_delta_r_in_real_space(delta_r = delta_r_data[-1],position_array=position_space_grid,momentum_array=momentum_space_grid,volume=volume,spin_index=True)
    
    function_for_plotting_c_c_dagger(np.reshape(Gamma_m_data[-1],(2*N_f,2*N_f)),N_f)
    function_for_plotting_c_dagger_c(np.reshape(Gamma_m_data[-1],(2*N_f,2*N_f)),N_f)
    function_for_plotting_c_dagger_c_dagger(np.reshape(Gamma_m_data[-1],(2*N_f,2*N_f)),N_f)
    function_for_plotting_c_c(np.reshape(Gamma_m_data[-1],(2*N_f,2*N_f)),N_f)    

    function_for_plotting_Gamma_b(np.reshape(Gamma_b_data[-1],(2*N_b,2*N_b)),N_b)