# -*- coding: utf-8 -*-
"""
Created on  Sep 13, 2024

@author: Yash_palan

File containing the class definition for the input variables and the computed variables class
"""
##############################################################################
##############################################################################
import numpy as np
import scipy.sparse as sp
import torch
##############################################################################
##############################################################################
# Define a class that can store all the input variabes
class input_variables:
    """
    Stores all the fixed input variables that we will need in the computation of the Hamiltonian.
    ....

    Attributes
    ----------

    fermionic_position_array: numpy array
                        Size : N_f x dim
                    The position space grid.

    bosonic_position_array: numpy array 
                            Size : N_b x dim
                            The momentum space grid.

    N_b: int
        Number of bosons in the system. 
    
    N_f: int
        Number of fermions in the system.   
    
    lmbda: numpy array
            Size : N_b x N_f
            The Lang Firsov transformation coefficient.

    J_0: float
        Size : N_f x N_f
        Holstein model Fermionic coupling strength
    
    gamma: numpy array
        Size : N_b x N_f
            The electron-phonon interaction strength with momentum (k).
    
    omega: numpy array
        Size : 1 x N_b
            The frequency of the bosons (omega) with momentum (k).

    fourier_array: numpy array
                    Size : N_b x N_f
                    The Fourier transform matrix e^{i q x}

    """
    def __init__(self,fermionic_position_array:np.ndarray,bosonic_position_array:np.ndarray,fourier_array:np.ndarray,
                lmbda:np.ndarray,J_0:np.ndarray,gamma:np.ndarray,omega:np.ndarray,chemical_potential_val=0.0):
        self.fermionic_position_array = fermionic_position_array
        self.bosonic_position_array = bosonic_position_array
        self.fourier_array = fourier_array
        self.N_b = len(bosonic_position_array)
        self.N_f = len(fermionic_position_array)
        self.lmbda = lmbda    # Size : N_b x N_f 
        self.J_0 = J_0
        self.gamma = gamma
        self.omega = omega
        self.chemical_potential_val = chemical_potential_val
    
    def updating_lambda_bar_from_lambda(self,lmbda_q:np.ndarray,volume:float,spin_index=True,update_in_input_variables=True)->np.ndarray:
        """
        This function computes the lambda_bar matrix from the lambda matrix. 
        This is done by taking the outer product of the lambda matrix with itself.

        Retruns
        -------
        TYPE: numpy array
        SIZE: N_b x N_f

        """
        
        if(spin_index==True):
            final_mat = 1/volume*np.einsum('q,qn,qm->nm',lmbda_q,self.fourier_array[:,0:self.N_b],np.conj(self.fourier_array))
        if(spin_index==False):
            final_mat = 1/volume*np.einsum('q,qn,qm->nm',lmbda_q,self.fourier_array,np.conj(self.fourier_array))
       
        if(update_in_input_variables==True):
            self.lmbda = final_mat
        
        return(final_mat)
    
    def updating_lambda_bar_from_spin_removed_lambda_bar(self,spin_removed_lambda_bar:np.ndarray,spin_index=True,update_in_input_variables=True)->np.ndarray:
        """
        This function computes the lambda_bar matrix from the lambda matrix. 
        This is done by taking the outer product of the lambda matrix with itself.

        Retruns
        -------
        TYPE: numpy array
        SIZE: N_b x N_f

        """

        # if(spin_index==True):
        #     final_mat = 1/volume*np.einsum('q,qn,qm->nm',lmbda_q,self.fourier_array[:,0:self.N_b],np.conj(self.fourier_array))
        # if(spin_index==False):
        #     final_mat = 1/volume*np.einsum('q,qn,qm->nm',lmbda_q,self.fourier_array,np.conj(self.fourier_array))
        if(spin_index==True):
            final_mat = np.append(spin_removed_lambda_bar,spin_removed_lambda_bar,axis=1)
        else:
            final_mat = spin_removed_lambda_bar
        
        if(final_mat.shape != (self.N_b,self.N_f)):
            raise Exception("The shape of the matrix is not correct. Check the code again")

        if(update_in_input_variables==True):
            self.lmbda = final_mat
        
        return(final_mat)
##############################################################################
##############################################################################

##############################################################################
##############################################################################
# Define a class that can store all the initial variabes that we need
class computed_variables:
    """
    This class stores all the values of the variables that we will need in definition of the Hamiltonian.
    
    ...
    
    Attributes
    ----------
    omega_bar_mat:  sparse array (2N_b,2N_b)
                    The onsite energy of the phonons (bosons) in the system. 

    J_i_j_mat: numpy array (N_f,N_f)
                Tunneling probability of the electrons from one site to another.

    alpha_bar_mat:  numpy array (N_f,N_f,2N_b)

    delta_gamma_tilde_mat: numpy array (2N_b,N_f)
                            Effective electron-phonon interaction matrix
                            
    Ve_i_j_mat: numpy array (N_f,N_f)
                Effective electron electron interaction  (4 point vertex)

    Methods
    -----------------------------
    -----------------------------
    omega_bar(input_variable)
        computes the matrix omega_bar_mat  

    alpha_bar(input_variable)
        compute the alpha_bar_mat
    
    J_i_j(input_variable,delta_r,Gamma_b)
        Compute the Tunneling matrix J_i_j_mat

    chemical_potential(input_variable)
        Returns the chemical potential matrix (This function defined the chemical potential)

    delta_gamma(input_variables)
        Compute the delta_gamma matrix

    delta_gamma_tilda(input_variables)
        Compute the delta_gamma_tilde matrix

    Ve_i_j(input_variables)
        Compute the Ve_i_j matrix
    -----------------------------

    """    
    def __init__(self,N_b:int,N_f:int):
        """
        -----------------------------
        -----------------------------
        DESCRIPTION
        -----------------------------
        -----------------------------
        This class stores all the values of the variables that we will need in the computation of the Hamiltonian.
        -----------------------------
        -----------------------------
        PARAMETERS
        -----------------------------
        -----------------------------
        N_b :   TYPE: int
                DESCRIPTION: The number of bosons in the system.

        N_f :   TYPE: numpy array
                DESCRIPTION: The number of fermions in the system.    

        """
        self.omega_bar_mat = torch.zeros((2*N_f,2*N_f),dtype=torch.complex128)
        self.J_i_j_mat = torch.zeros((N_f,N_f),dtype=torch.complex128)
        self.alpha_bar_mat = torch.zeros((N_f,N_f,2*N_b),dtype=torch.complex128)
        self.delta_gamma_mat = torch.zeros((N_b,N_f),dtype=torch.complex128)
        self.delta_gamma_tilde_mat = torch.zeros((2*N_b,N_f),dtype=torch.complex128)   
        self.Ve_i_j_mat = torch.zeros((N_f,N_f),dtype=torch.complex128)
    
    ##############################################################################
    ##############################################################################
    # Remarks    
    # 6/9/24 : Checked all of the functions below in this class again using MATLAB code and they seem to be working fine.
    ##############################################################################
    ##############################################################################
    # Functions needed to describe the Phonon self energy of the Hamiltonian
    def omega_bar(self,input_variable:input_variables)->torch.Tensor:
        # Check if this gives the correct ouput
        # Gives the correct output : Yash 10/5/2024 12:10
        """
        ----------
        ----------
        Description
        ----------
        ----------
        This function creates the omega_bar matrix, which stores the Phonon frequencies for the quadrature operators.
        This comes from the KE term for the bosons in the Hamiltonian. 
        
        This can be seen in the writeup as the term ________.

        -----------------------------
        -----------------------------
        Parameters:
        omega:  TYPE : numpy array
                SIZE : N_b x N_b
                DESCRIPTION: This is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
        data_type:  TYPE: String
                    DESCRIPTION: This defines the data type of the final matrix.
        -----------------------------
        -----------------------------       
        Return :    TYPE : complex torch tesnor
                    SIZE : 2N_b x 2N_b
                    DESCRIPTION: This is the final matrix that is returned. which stores the information about omega_bar.
        """
        if(isinstance(input_variable, input_variables)==False):
            raise Exception("The input variables are not of the correct type")
        
        omega = input_variable.omega
        self.omega_bar_mat = torch.kron(torch.eye(2),torch.tensor(omega,dtype=torch.complex128))
        np.save("omega_bar_mat.npy",self.omega_bar_mat.numpy())
        return(self.omega_bar_mat)
    
    ##############################################################################
    ##############################################################################
    # Functions needed to describe the Electronic tunneling part of the Hamiltonian
    
    def alpha_bar(self,input_variable:input_variables)->torch.Tensor:
        # Check if this gives the correct ouput
        # Gives the correct output : Yash 10/5/2024 16:04

        """
        -----------------------------
        -----------------------------
        DESCRIPTION
        -----------------------------
        -----------------------------
        This function creates the alpha_{i,j} vector (in the writeup). This determines the tunneling 
        probability of the electrons from one site to another. When exponentiated, this gives the 
        tunneling probability of the electrons.
        
        -----------------------------
        -----------------------------
        PARAMETERS
        -----------------------------
        -----------------------------
        0<= i,j < N_f
        
        i : TYPE: int
            DESCRIPTION: This is the index of the 1st Fermion
        j : TYPE : int
            DESCRIPTION: This is the index of the 2nd Fermion
            
        lmbda:  TYPE - numpy array
                SIZE - 1 x N_b
                DESCRIPTION: This is the variation of lambda(k), the Lang Firsov transformation coefficient.
        momentum_array: type - numpy array
                        tells us the momentum space grid.
        position_array: type - numpy array
                        tells us the real space grid
        volume: type - float
                        Volume of the system, as used in the Hamiltonian.
        N_b: type - float
                        Number of bosons, or basically the number of momentum points in the k grid 
                        (as the number of bosons is given by the discretisation of the bosonic annihilation 
                        operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
                        momentum grid).
        
        -----------------------------
        -----------------------------
        Return: TYPE - complex pytorch tensor
                SIZE - N_f x N_f x 2N_b
                DESCRIPTION: This is the final matrix that is returned.
        """
        if(isinstance(input_variable, input_variables)==False):
            raise Exception("The input variables are not of the correct type")

        N_f = input_variable.N_f
        N_b = input_variable.N_b
        lmbda = input_variable.lmbda

        # Check that this works properly
        lambda_temp = np.transpose(lmbda)
        temp_mat = np.reshape(lambda_temp,(N_f,1,N_b))- np.reshape(lambda_temp,(1,N_f,N_b))
        final_mat = np.append(np.zeros((N_f,N_f,N_b)),temp_mat ,axis=2)

        if(final_mat.shape != (N_f,N_f,2*N_b)):
            raise Exception("The shape of the matrix is not correct. Check the code again")
        
        # Converting to a pytorch tensor 
        self.alpha_bar_mat = torch.tensor(final_mat,dtype=torch.complex128)
        np.save("alpha_bar_mat.npy",self.alpha_bar_mat.numpy())
        return(self.alpha_bar_mat)
    
    def J_i_j(self,input_variable:input_variables,delta_r:np.ndarray,Gamma_b:np.ndarray)->torch.Tensor:
        # Check if this gives the correct ouput
        # Seems to give the correct output : 
        # This is because the only change that I made to the above is convert all numpy function to torch function and
        # have checked that the final pytorch tensor is of the correct type and size.
        # Yash 15/07/2024 10:41
        
        """
        DESCRIPTION of function
        -----------------------------
        -----------------------------
        J_i_j is the term which comes out ot the electronic KE and depends on alpha_{i,j}, since
        the tunneling probability of the electrons from one site to another is determined by alpha_{i,j}.
        
        This function computes the expectation of the operator J_{i,j} defined in 
        the writeup. So, its <J_{i,j}>_{GS} and NOT J_{i,j} operator. 

        -----------------------------
        -----------------------------
        INPUT Variables decription
        -----------------------------
        -----------------------------
        0<= i,j < N_f
        
        J_0 :   TYPE : real float / hermitian N_f x N_f matrix
                DESCRIPTION: Holstein model Fermionic coupling strength
        lmbda:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION: this is the variation of lambda(k), the Lang Firsov transformation coefficient.
        momentum_array: TYPE: numpy array
                        SIZE: 1 x N_b
                        DESCRIPTION: Tells us the momentum space grid.
        position_array: TYPE: numpy array
                        SIZE: 1 x N_f
                        DESCRIPTION: tells us the real space grid
        volume: TYPE : float
                DESCRIPTION: Volume of the system, as used in the Hamiltonian.
        N_b:    TYPE : float
                DESCRIPTION: Number of bosons, or basically the number of momentum points in the k grid 
                        (as the number of bosons is given by the discretisation of the bosonic annihilation 
                        operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
                        momentum grid).
        
        -----------------------------
        -----------------------------
        RETURNS
        -----------------------------
        -----------------------------
        Size - N_f x N_f
        TYPE - complex pytorch tensor 
        DESCRIPTION - Computes the Tunneling matrix J_i_j_mat for the transformed Hamiltonian.
            
        """
        # Just to check if the user has entered the correct type of input variables
        if(isinstance(input_variable, input_variables)==False):
            raise Exception("The input variables are not of the correct type")
        
        N_f= input_variable.N_f
        J_0 = torch.from_numpy(input_variable.J_0).to(dtype=torch.complex128)
        alpha_i_j_mat = self.alpha_bar_mat


        # Just to see that if J_0 is a constant, implying that the tunneling probability is the same for all the sites and hence 
        # we just multiply with np.ones((N_f,N_f)) to get the final matrix J_0 in a matrix form.
        if(isinstance(J_0,int) or isinstance(J_0,float) or isinstance(J_0,complex) ):
            J_0 = J_0*torch.ones((N_f,N_f)).to(dtype=torch.complex128)

        matrix_1 = torch.einsum('k,ijk->ij',torch.tensor(delta_r,dtype=torch.complex128),alpha_i_j_mat)
        matrix_2 = torch.einsum('ijk,kl,ijl->ij',alpha_i_j_mat,torch.tensor(Gamma_b,dtype=torch.complex128),alpha_i_j_mat)

        try:
            self.J_i_j_mat = J_0*torch.exp(-1j*matrix_1)*torch.exp(-0.5*matrix_2)        
            if(torch.any(torch.isnan(self.J_i_j_mat))):
                raise Exception("There is a NaN in the computation of the J_i_j matrix. Figure out why this is happening.")
        except OverflowError:
            raise Exception("There is an overflow error comming in the computation of the J_i_j matrix. Figure out why this is happening.")
        except:
            raise Exception("There is an error in the computation of the J_i_j matrix")
        np.save("J_i_j_mat.npy",self.J_i_j_mat.numpy())
        return(self.J_i_j_mat) 
    
    def chemical_potential(self,input_variable:input_variables)->torch.Tensor:

        """
        Returns the chemical potential matrix (This function defined the chemical potential)

        -----------------------------
        RETURNS
        -----------------------------
        -----------------------------
        final_matrix:    TYPE: float pytorch tensor
                        SIZE: N_f x N_f
                        DESCRIPTION: This is the final matrix that is returned.
        """
        ## Should be given as an input at the start of the main code.
    
        N_f = input_variable.N_f
        final_matrix = torch.from_numpy( np.diag(input_variable.chemical_potential_val*np.ones((N_f,)) - 0.5*np.diag(self.Ve_i_j_mat) ) )
        return final_matrix
    
    ##############################################################################
    ##############################################################################
    # Functions needed to describe the electron-phonon coupling part of the Hamiltonian
    def delta_gamma(self,input_variables:input_variables)->np.ndarray:
        # Works properly: Yash 10/5/24 16:22
        """
        DESCRIPTION
        -----------------------------
        -----------------------------
        This function creates the delta_gamma matrix (in the writeup). 
        This is basically the renormalised electron-phonon coupling matrix.
        However, this is the matrix when we have defined the Physical position and momnetum operators in the 
        Fourier space. This will be converted to the new matrix delta_gamma_tilde, which is what we 
        will generally use in the code.
        -----------------------------
        -----------------------------   
        PAAMETERS
        -----------------------------
        -----------------------------
        omega:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION:this is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
        gamma:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION: this is the variation of gamma(k), the electron-phonon interaction strength with momentum (k).
        lmbda:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION: this is the variation of lambda(k), the Lang Firsov transformation coefficient.
        momentum_array: TYPE: numpy array
                        SIZE: 1 x N_b
                        DESCRIPTION:tells us the momentum space grid.
        position_array: TYPE: numpy array
                        SIZE: 1 x N_f
                        DESCRIPTION: tells us the real space grid
        volume: TYPE: float
                DESCRIPTION: Volume of the system, as used in the Hamiltonian.
        N_b:    TYPE: float
                DESCRIPTION: Number of bosons
        N_f:    TYPE: float
                DESCRIPTION:Number of fermions
        -----------------------------
        -----------------------------
        RETRUNS
        -----------------------------
        -----------------------------
        final_array:    TYPE: complex numpy array
                        SIZE: N_b x N_f
                        DESCRIPTION: This is the final matrix that is returned.
        """
        N_b = input_variables.N_b
        N_f = input_variables.N_f
        gamma = input_variables.gamma
        lmbda = input_variables.lmbda
        omega = input_variables.omega

        self.delta_gamma_mat = gamma - np.einsum('kl,lj->kj',omega,lmbda) 

        # Just a check to see if the final array has the correct shape
        if(np.shape(self.delta_gamma_mat)!=(N_b,N_f)):
            raise Exception("The shape of the matrix is not correct. Check the code again")
        np.save("delta_gamma_mat.npy",self.delta_gamma_mat)
        return (self.delta_gamma_mat)

    def delta_gamma_tilda(self, input_variables:input_variables)->torch.Tensor:

        # Check if this gives the correct ouput
        # First created : Yash 16/4/2024 12:21
        
        """
        -----------------------------
        -----------------------------
            DESCRIPTION
        -----------------------------
        -----------------------------
        This function creates the delta_gamma_tilde matrix (in the writeup). 

        This is basically the renormalised electron-phonon coupling matrix.
        However, this is the matrix when we have defined the Hermitian position and momnetum operators in the 
        Fourier space. 
        This is what will generally be used in the code for computation purposes.
        
        -----------------------------
        -----------------------------
        PARAMETERS
        -----------------------------
        -----------------------------

        omega:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION:this is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
        gamma:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION: this is the variation of gamma(k), the electron-phonon interaction strength with momentum (k).
        lmbda:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION: this is the variation of lambda(k), the Lang Firsov transformation coefficient.
        momentum_array: TYPE: numpy array
                        SIZE: 1 x N_b
                        DESCRIPTION:tells us the momentum space grid.
        position_array: TYPE: numpy array
                        SIZE: 1 x N_f
                        DESCRIPTION: tells us the real space grid
        volume: TYPE: float
                DESCRIPTION: Volume of the system, as used in the Hamiltonian.
        N_b:    TYPE: float
                DESCRIPTION: Number of bosons
        N_f:    TYPE: float
                DESCRIPTION:Number of fermions
        -----------------------------
        -----------------------------
        RETRUNS
        -----------------------------
        -----------------------------
        final_array: TYPE: complex pytorch tensor
            SIZE: 2*N_b x N_f
            DESCRIPTION: This is the final matrix that is returned.    
        """
        N_b = input_variables.N_b
        N_f = input_variables.N_f
        # Converting to a pytorch tensor for speed in computations
        self.delta_gamma_tilde_mat = torch.from_numpy(np.append(self.delta_gamma_mat,np.zeros((N_b,N_f)),axis=0) ).to(dtype=torch.complex128)
        
        if(np.shape(self.delta_gamma_tilde_mat) != (2*N_b,N_f)):
            raise Exception("The shape of the matrix is not correct. Check the code again")
        
        return (self.delta_gamma_tilde_mat)
    ##############################################################################
    ##############################################################################
    # Functions needed to describe the effective electron-electron interaction part of the Hamiltonian
    def Ve_i_j(self, input_variables:input_variables)->torch.Tensor:
        # Check if this gives the correct ouput
        # Working propertly : Yash 10/5/24 16:47
        # Update : Yash 5/6/24 : Changed the definition of Ve to the one that is similar to Shi et. al. for better comparison.
        """
        -----------------------------
        -----------------------------
        DESCRIPTION
        -----------------------------
        -----------------------------
        This function creates the matrix for the effective electron-electron interaction term.

        -----------------------------
        -----------------------------
        PARAMETERS
        -----------------------------
        -----------------------------
        gamma:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION: this is the variation of gamma(k), the electron-phonon interaction strength with momentum (k).
        volume: TYPE: float
        momentum_array: TYPE: numpy array
                        SIZE: 1 x N_b
                        DESCRIPTION: tells us the momentum space grid.
        position_array: TYPE: numpy array
                        SIZE: 1 x N_f
                        DESCRIPTION: tells us the real space grid
        N_b:    TYPE: float
                DESCRIPTION: Number of bosons
        N_f:    TYPE: float
                DESCRIPTION:Number of fermions
        lmbda:  TYPE: numpy array
                SIZE: 1 x N_b
                DESCRIPTION: this is the variation of lambda(k), the Lang Firsov transformation coefficient.
        omega:  TYPE - numpy array
                SIZE: 1 x N_b
                this is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
        -----------------------------
        -----------------------------
        RETURNS:
        -----------------------------
        -----------------------------
        final_array: TYPE: complex pytorch tensor
                SIZE: N_f x N_f
                DESCRIPTION: This is the final matrix that is returned.
        -----------------------------
        -----------------------------
        
        """
        gamma = input_variables.gamma
        lmbda = input_variables.lmbda
        omega = input_variables.omega

        temp_mat_1 = np.einsum('ki,kl,lj->ij',lmbda,omega,lmbda)
        temp_mat_2 = np.einsum('ki,kj->ij',gamma,lmbda)
        self.Ve_i_j_mat = torch.tensor(  2.0*(temp_mat_1 - temp_mat_2 - temp_mat_2.T )  ,  dtype=torch.complex128)
        np.save("Ve_i_j_mat.npy",self.Ve_i_j_mat.numpy())
        return self.Ve_i_j_mat
    
    def initialize_all_variables(self,input_variables:input_variables,delta_r:np.ndarray,Gamma_b:np.ndarray):
        self.omega_bar(input_variables)
        self.alpha_bar(input_variables)
        self.delta_gamma(input_variables)
        self.delta_gamma_tilda(input_variables)
        self.Ve_i_j(input_variables)
        self.J_i_j(input_variables,delta_r,Gamma_b)

##############################################################################
##############################################################################


##############################################################################
##############################################################################