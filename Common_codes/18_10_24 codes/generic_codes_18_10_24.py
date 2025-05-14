# -*- coding: utf-8 -*-
"""
Created on  Sep 13, 2024

@author: Yash_palan

This file just stores codes that are extremely generic like saving data to a file,
or making the sigma matrix, delta matrix, and coordinate arrays etc.


------------------------------------------
------------------------------------------

Comparison with the 11/7/24 version:
This one assumes that the input variables gamma_k and lmbda_k are normalised with the volume, 
i.e. in terms of the old notation, we have 
lmbda_{in this file} = lmbda/sqrt(volume)
gamma_{in this file} = gamma/sqrt(volume)

due to which we do not need to keep the volume factor everywhere, which hopefully makes the 
RK4 method more stable.
"""
##############################################################################
##############################################################################
# from statistics import correlation
import numpy as np
import csv   
##############################################################################
##############################################################################

def save_data_to_file(file_name,data):
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    This function saves the data to a file. 

    -----------------------------
    -----------------------------
    PARAMETERS
    -----------------------------
    -----------------------------
    file_name : TYPE : String
                DESCRIPTION : This is the name of the file where we want to save the data.
    data : TYPE : numpy array
                DESCRIPTION : This is the data that we want to save in the file.
    -----------------------------
    -----------------------------
    RETURNS
    -----------------------------
    -----------------------------
    """
    # with open(file_name, 'a') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)

    file = open(file_name, 'a')
    writer = csv.writer(file)
    writer.writerows(np.reshape(data,(1,-1)))
    file.close()
    return

def save_data_to_file_2(file_name,data):
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    This function saves the data to a file. 
    This is just so that I can save the final results of the ODE solver.
    -----------------------------
    -----------------------------
    PARAMETERS
    -----------------------------
    -----------------------------
    file_name : TYPE : String
                DESCRIPTION : This is the name of the file where we want to save the data.
    data : TYPE : numpy array
                DESCRIPTION : This is the data that we want to save in the file.
    -----------------------------
    -----------------------------
    RETURNS
    -----------------------------
    -----------------------------
    """
    file = open(file_name, 'a')
    writer = csv.writer(file)

    
    if(type(data)==np.ndarray):
        if(data.ndim == 1):
            writer.writerows([data.tolist()])
        if(data.ndim > 1 ):
            writer.writerows([data.tolist()])
    else:
        writer.writerows(data)
    file.close()
    return

##############################################################################
##############################################################################
# Define basic variables that are needed everywhere in the computations
def sigma(N_b,dtype="float"):
    # Check if this gives the correct ouput
    # Gives the correct output : Yash 6/10/2023 12:04
    """
    This function creates the sigma matrix. 
    The sigma matrix is the matrix which is used in the definition of a Symplectic matrix.

    Parameters:

    N_b:    TYPE: int
            SIZE: 1
            DESCRIPTION: Number of bosons, or basically the number of momentum points in the k grid 
                        (as the number of bosons is given by the discretisation of the bosonic annihilation 
                        operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
                        momentum grid).

    Returns: 
    Size: 2N_b x 2N_b
    Desceiption: Returns the Symplectic sigma/Omega matrix (which is 2N_b x 2N_b matrix)
    
    """
    mat_1 = np.array([[0.0,1.0],[-1.0,0.0]])
    mat_2 = np.identity(N_b)
    final_mat=np.kron(mat_1, mat_2)
    final_mat = final_mat.astype(dtype)
    return(final_mat)

def delta(i,j):

    """
    ----------
    ----------
    DESCRIPTION
    ----------
    ----------
    Just our favourite Kronecker delta function     
    
    ----------
    ----------   
    Parameters
    ----------
    ----------

    i : TYPE: int     
    j : TYPE: int
    
    ----------
    ----------
    Returns
    ----------
    ----------
    TYPE: int
    """
    if(i==j):
        return 1
    else:
        return 0

# Works fine : Yash 13/9/24
def coordinate_array_creator_function(L_min,L_max,number_of_points,spin_index=True,dtype="float"):

    """
    ----------
    ----------
    Description
    ----------
    ----------    
    Creates the array for the momentum space and the position space grid. 
    Given the lower limit value and upper limit values, this function creates the grid of points
    for each dimension.
    
    Remark - Now, depending on whether we have spin or not, things will be a little different.
    
    The Idea is to consider the notation c_{i} where now i->(i_p,i_s) and hence encode everyting
    into one index. So, effectively, in language of arrays, we have doubled the number of fermions
    with the difference being that now the position array will have say N_f/2 points 
    and hence to accomodate for the other spins, we will have to double the position array. 
    
    However this does not create any issues with the scenario when we have to deal with many 
    dimensions.

    ----------
    ----------
    Parameters
    ----------
    ----------
    L_max:
    Type: numpy array
    Size: 1 X dim (dim,)
    Despcription: This is the upper limit of the grid for each dimension.

    L_min:
    Type: numpy array
    Size:  1 X dim (dim,)
    Despcription: This is the lower limit of the grid for each dimension.

    number_of_points:
    Type: numpy array
    Size:  1 X dim (dim,)
    Despcription: This is the number of points in the grid for each dimension.

    spin_index:
    Type: bool
    Description: This is a boolean value which tells us whether we need to consider the spin index or not.

    dtype:
    Type: String
    Description: This defines the data type of the final matrix.
    
    ----------
    ----------   
    Returns
    ----------
    ----------
    TYPE: numpy array
    SIZE: if (spin index==0)
            1 x (multiplication of number_of_points in each dimension) 
          else
            1 x 2*(multiplication of number_of_points in each dimension)
        
        However, for either the momentum or position coordinates, the above size can be easily said to be 
            1 x N_b or 1 x N_f respectively.

    DESCRIPTION: This is the final matrix that stores all the coordinate values.

    """
    # Checking if the input arrays are 1D arrays and have the same length
    if(len(np.shape(L_max))>1 or len(np.shape(L_min))>1):
        print("L_max and L_min arrays are not 1D arrays")
    elif(np.size(L_max)!=np.size(L_min)):
        print("L_max and L_min arrays do not have the same lengths") 
    elif( np.size(number_of_points)!=1 and np.size(number_of_points)!=np.size(L_max)):
        print("number_of_points array is not a 1D array or does not have the same length as L_max and L_min")   
    else:
        # Create a slice object for each dimension
        if(np.size(number_of_points)==1 and np.size(L_max)>1):
            number_of_points = np.tile(number_of_points,np.size(L_min))

        if(np.size(L_max)==1):
            coordinate_array =np.linspace(  L_min,  L_max,  number_of_points,endpoint=False  )
        
        else:
            slices = [slice(  L_min[dim],  L_max[dim],  (L_max[dim]-L_min[dim])/number_of_points[dim]  ) for dim in range(len(L_max))]
            
            # Use mgrid to generate the grid of points
            coordinate_array = np.mgrid[slices].reshape(len(L_max), -1).T

    

    # Deals with the fact that we have spin up and spin down fermions. Note that we only consider spin
    # 1/2 particles and nothing more than that. Can be generalised for higher spins as well.
    if(spin_index==True):
        coordinate_array = np.append(coordinate_array,coordinate_array,axis=0)

    return coordinate_array

# Need to check this one. All others seem to be working fine.
def creating_J_0_matrix(position_array:np.ndarray,J_0,Length,spin_index=True)->np.ndarray:
    """
    This function creates the J_0 matrix. 
    The J_0 matrix is the matrix which is used in the definition of the tunneling matrix J_{i,j}.

    Parameters:

    N_f:    TYPE: int
            DESCRIPTION: Number of fermions in the system. 

    J_0:    TYPE: real float / hermitian N_f x N_f matrix
            DESCRIPTION: Holstein model Fermionic coupling strength
        """
    N_f = position_array.shape[0]
    nearest_neighbour_vectors = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    # nearest_neighbour_vectors = np.array([1,-1])

    if(spin_index==True):
        size_of_individual_block = int(N_f/2)
    else:
        size_of_individual_block = N_f
        
    J_0_mat_block = np.zeros((int(size_of_individual_block),int(size_of_individual_block))) 
    for i in range(size_of_individual_block):
        for j in range(size_of_individual_block):
            for ele in nearest_neighbour_vectors:
                if(False in set (np.mod(position_array[i]+ele,Length)==position_array[j])):
                    continue
                J_0_mat_block[i,j] = J_0
    if(spin_index==True):
        J_0_matrix = np.kron([[1,0],[0,1]],J_0_mat_block)
    else:
        J_0_matrix = J_0_mat_block
    return(J_0_matrix)

def creating_intial_random_lambda_q_matrix_2D(momentum_array:np.ndarray,rep_val_x,rep_val_y,factor = 1)->np.ndarray:
    """
    Description
    -----------
    The point of this function is to create a random lambda_q matrix which has the property that lambda_q = lambda^{*}_{-q}.
    Here we just choose lambda_q to be real for simplicity, so that life is easy. However, we can choose it to be complex as well, but 
    for now its real.


    Parameters:
    -----------
    momentum_array: TYPE: numpy array
                    SIZE: N_b x dim (dim = 1D/2D/3D, so 1,2 or 3)
                    DESCRIPTION: This is the momentum array that we have.
    rep_val_x: TYPE: real float
                DESCRIPTION: This is the value of the periodicity in the x direction (this should be pi/a, a being the lattice length, however.
                since we don't want to pass that, we just assume the user will give that as an input).
    rep_val_y: TYPE: real float
                DESCRIPTION: This is the value of the periodicity in the y direction.
    factor: TYPE: real float
            DESCRIPTION: This is the factor by which we multiply the random number.
    
    Returns
    -------
    TYPE: numpy array
    SIZE: 1 x N_b
    DESCRIPTION: This is the initial random lambda_q matrix. 
    """
    if(  np.mod (momentum_array.shape[0]-4,2)!=0 ):
        raise Exception("The number of momentum points is not correct")
    
    N_b = momentum_array.shape[0]
    lambda_q = np.zeros((int(N_b),),dtype='float')

    if(len(lambda_q)!= momentum_array.shape[0]):
        raise Exception("The length of the lambda_q array is not correct")
    
    for i in range(len(lambda_q)):
        kx_val = momentum_array[i, 0]
        ky_val = momentum_array[i, 1]

        kx_min, ky_min = np.min(momentum_array,axis=0)
        kx_max , ky_max = np.max(momentum_array,axis=0)

        # rep_val_x = kx_max - kx_min + momentum_array[0,0] 
        # rep_val_y = ky_max - ky_min
        if(np.all(momentum_array[i] == [0,0]) or np.all(momentum_array[i] == [-np.pi,-np.pi]) or np.all(momentum_array[i] == [0,-np.pi]) or
           np.all(momentum_array[i] == [-np.pi,0]) ):
            lambda_q[i] = np.random.rand()
            continue

        # Since we choose the region -pi to pi, we know that the negative will also lie inside the region and hence dont have 
        # to worry about the negative values, except for the cases when we are on the boundaries
        ix_conj = -kx_val 
        iy_conj = -ky_val 

        # To deal with boundaries of the BZ (like the line kx = -pi, ky = -pi , where the negative goes to pi)
        if(ix_conj > kx_max):
            ix_conj = ix_conj - rep_val_x
        if(iy_conj > ky_max):
            iy_conj = iy_conj - rep_val_y
        if(ix_conj<kx_min):
            ix_conj = ix_conj + rep_val_x
        if(iy_conj<ky_min):
            iy_conj = iy_conj + rep_val_y
        
        conj_loc = np.where(np.all(momentum_array == [ix_conj,iy_conj], axis=1))
        lambda_q[i] = np.random.rand()
        lambda_q[conj_loc] = lambda_q[i] 
    
    return(lambda_q)

def creating_fourier_matrix(momentum_space_array:np.ndarray,position_space_array:np.ndarray):
    """
    Parameters:
    -----------
    momentum_space_array: TYPE: numpy array
                            SIZE: N_b x dim
                            DESCRIPTION: This is the momentum space grid.
    position_space_array: TYPE: numpy array
                            SIZE: N_f x dim
                            DESCRIPTION: This is the position space grid

    Returns :
    ---------
    TYPE : numpy array
    SIZE : N_b x N_f
    DESCRIPTION : This is the Fourier matrix that we need to create.
                    e^{i q m}
    """
    if(len(position_space_array.shape)==1):
        fourier_sapce_matrix = np.exp(1j*np.einsum('q,m->qm',momentum_space_array,position_space_array))
    if(len(position_space_array.shape)>1):
        fourier_sapce_matrix = np.exp(1j*np.einsum('ql,ml->qm',momentum_space_array,position_space_array))
    return(fourier_sapce_matrix)

def creating_fourier_matrix_fermions(momentum_space_array_fermions:np.ndarray,position_space_array:np.ndarray):
    """
    Parameters:
    -----------
    momentum_space_array: TYPE: numpy array
                            SIZE: N_f x dim
                            DESCRIPTION: This is the momentum space grid.
    position_space_array: TYPE: numpy array
                            SIZE: N_f x dim
                            DESCRIPTION: This is the position space grid

    Returns :
    ---------
    TYPE : numpy array
    SIZE : N_f x N_f
    DESCRIPTION : This is the Fourier matrix that we need to create.
                    e^{i q m}
    """
    if(len(position_space_array.shape)==1):
        fourier_sapce_matrix = np.exp(1j*np.einsum('q,m->qm',momentum_space_array_fermions,position_space_array))
    if(len(position_space_array.shape)>1):
        fourier_sapce_matrix = np.exp(1j*np.einsum('ql,ml->qm',momentum_space_array_fermions,position_space_array))
    return(fourier_sapce_matrix)