# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 18:52:28 2023

@author: Yash_palan
This file is used for definining all the Real time functions that we will need.

"""

##############################################################################
##############################################################################
from Common_codes import global_functions_v2 as gf
import numpy as np
import math as mt
import cmath
import itertools as it
##############################################################################
##############################################################################
# First equation of motion functions

def first_eq_term_1(delta_R,Gamma_b,Gamma_f,N_b,lmbda,momentum_array,position_array,volume,N_f,J_0):
    """
    Parameters
    ----------
    delta_R : TYPE
        DESCRIPTION.
    Gamma_b : TYPE
        DESCRIPTION.
    Gamma_f : TYPE
        DESCRIPTION.
    N_b : TYPE
        DESCRIPTION.
    lmbda : TYPE
        DESCRIPTION.
    momentum_array : TYPE
        DESCRIPTION.
    position_array : TYPE
        DESCRIPTION.
    volume : TYPE
        DESCRIPTION.
    N_f : TYPE
        DESCRIPTION.
    J_0 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """    
    Omega_mat_real = gf.Omega(N_b)
    Omega_mat = Omega_mat_real.astype(complex)
    
    #Making the matrix where the values get stored
    final_mat = np.zeros(2*N_b,dtype=complex)
    
    

    # Note - Works Properly 10/10/2023 12:38
    for j in range(2*N_b):
        for ip in range(N_f):
            for jp in range(N_f):
                alpha_ip_jp_mat = gf.alpha(ip,jp,N_b,lmbda,momentum_array,position_array,volume).astype(complex)       
                mat = np.matmul(alpha_ip_jp_mat,Omega_mat)  # just check this once more
                
                value_1= mat[j]
                value_2= cmath.exp( 1j*np.matmul(delta_R,alpha_ip_jp_mat) ) * cmath.exp( -0.5*np.matmul(alpha_ip_jp_mat ,np.matmul(Gamma_b,alpha_ip_jp_mat) ) ) 

                for sigma in range(2):
                    for sigmap in range(2):
        
                        value_3 = (2*gf.delta(ip,jp)*gf.delta(sigma,sigmap) \
                                 -1j*(Gamma_f[ip,jp,sigma,sigmap]+Gamma_f[ip+N_f,jp+N_f,sigma,sigmap]) \
                                     + (Gamma_f[ip,jp+N_f,sigma,sigmap] - Gamma_f[ip+N_f,jp,sigma,sigmap] ) )            
                        final_mat[j] += -J_0*0.5*value_1*value_2*value_3 
    return(final_mat)
    
    
    
# def first_eq_term_2(delta_R,omega,N_b):
#     """
#     delta_R: type - numpy complex array
#             size -1x2*N_b
#             This stores the expectatin values of all the Bosonic operators. 
    
#     omega: type - numpy array
#             this is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
#     N_b: type - float
#                     Number of bosons, or basically the number of momentum points in the k grid 
#                     (as the number of bosons is given by the discretisation of the bosonic annihilation 
#                     operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
#                     momentum grid).
#      """
#     # Note - Works Properly 6/10/2023 14:45
     
#     Omega_mat_real = gf.Omega(N_b)
#     omega_bar_mat_real = gf.omega_bar(omega)
   
#     # Converting all the above ,atrices to complex types
#     Omega_mat = Omega_mat_real.astype(complex)
#     omega_bar_mat = omega_bar_mat_real.astype(complex)
    
#     final_mat = 1j*np.matmul( Omega_mat,np.multiply(omega_bar_mat,delta_R) )          # np.multiply does element wise multiplication
#                                                                                              # np.matmul does matrix multiplication
#     return(final_mat)
def first_eq_term_2(delta_R,omega,N_b):
    """
    delta_R: type - numpy complex array
            size -1x2*N_b
            This stores the expectatin values of all the Bosonic operators. 
    
    omega: type - numpy array
            this is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
    N_b: type - float
                    Number of bosons, or basically the number of momentum points in the k grid 
                    (as the number of bosons is given by the discretisation of the bosonic annihilation 
                    operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
                    momentum grid).
     """     
    Omega_mat_real = gf.Omega(N_b)
    omega_bar_mat = gf.omega_bar(omega,"complex")
    omega_bar_mat_transpose = omega_bar_mat.transpose()
    
    # Converting all the above ,atrices to complex types
    Omega_mat = Omega_mat_real.astype(complex)
    
    
    final_mat = -(1j/2)*(-np.matmul(Omega_mat,omega_bar_mat.dot(delta_R)) +
                          +np.matmul(delta_R,omega_bar_mat_transpose.dot(Omega_mat)))                       
    return(final_mat)



def first_eq_term_3(Gamma_f,N_b,N_f,gamma,lmbda,omega,momentum_array,position_array,volume):
    """
    Gamma_f: type - numpy complex array
            size - N_f x N_f x 2 x 2
    delta_R: type - numpy complex array
            size -1x2*N_b
            This stores the expectatin values of all the Bosonic operators. 
    
    omega: type - numpy array
            this is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
    N_b: type - float
                    Number of bosons, or basically the number of momentum points in the k grid 
                    (as the number of bosons is given by the discretisation of the bosonic annihilation 
                    operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
                    momentum grid).
     """
    # Note - Works Properly 6/10/2023 14:45

    # Making real matrices 
    eta_mat_real = gf.eta(N_b,N_f,gamma,lmbda,omega,momentum_array,position_array,volume)
    Omega_mat_real = gf.Omega(N_b)
    
    # Converting all the above ,atrices to complex types
    Omega_mat = Omega_mat_real.astype(complex)
    eta_mat = eta_mat_real.astype(complex)
    
    #Making the matrix where the values get stored
    final_mat = np.zeros(2*N_b,dtype=complex) 
    
    # Computing the values of the final_mat matrix with contribution from the third term
    # i.e. from eqn 1.8 in the Git hub writeup names "Ch Summary of equations".
    for i in range(2*N_b):
        temp=0;
        for ip in range(2*N_b):
            for j in range(N_f):
                for sigma in range(2):
                    temp = temp+ (-1j)*Omega_mat[ip,i]*eta_mat[ip,j]*(1+ Gamma_f[j,j+N_f,sigma,sigma])    
        final_mat[i] = temp
    return(final_mat)
    
##############################################################################
##############################################################################
# Second equation of motion functions

def second_eq_term_1(delta_R,Gamma_b,Gamma_f,N_b,lmbda,momentum_array,position_array,volume,N_f,J_0):
    # Calculation is correct : Yash Palan 10/10/2023 12:38
    """
    -------------------------------------------------
    Remarks
    -------------------------------------------------
    Checked: Works properly,  Yash Palan 10/10/2023 12:38
    """ 
    Omega_mat_real = gf.Omega(N_b)
    Omega_mat = Omega_mat_real.astype(complex)
    
    #Making the matrix where the values get stored
    final_mat = np.zeros([2*N_b, 2*N_b],dtype=complex)
    for i in range(2*N_b):
        for j in range(2*N_b):
            temp=0
            for l in range(N_f):
                for m in range(N_f):
                    temp=0
                    alpha_ip_jp_mat_real = gf.alpha(l,m,N_b,lmbda,momentum_array,position_array,volume)        
                    alpha_ip_jp_mat = alpha_ip_jp_mat_real.astype(complex) 
                    
                    # Check the following line
                    value_1 = cmath.exp(1j*np.matmul(alpha_ip_jp_mat,delta_R)-0.5* np.matmul(alpha_ip_jp_mat,np.matmul(Gamma_b,alpha_ip_jp_mat)) )
                    
                    # Check the following line
                    value_2 = (1j)* ( np.matmul(Gamma_b,alpha_ip_jp_mat)[j]*np.matmul(alpha_ip_jp_mat,Omega_mat)[i] \
                                   - np.matmul(Gamma_b,alpha_ip_jp_mat)[i]*np.matmul(Omega_mat,alpha_ip_jp_mat)[j])
                    for sigma in range(2):
                        for sigmap in range(2):
                            ip=l
                            jp=m
                            # Check the folllowing lines
                            value_3 = (2*gf.delta(ip,jp)*gf.delta(sigma,sigmap) \
                                     -1j*(Gamma_f[ip,jp,sigma,sigmap]+Gamma_f[ip+N_f,jp+N_f,sigma,sigmap]) \
                                         + (Gamma_f[ip,jp+N_f,sigma,sigmap] - Gamma_f[ip+N_f,jp,sigma,sigmap] ) )            
                            final_mat[i,j] += -J_0*value_1*value_2*value_3         
    return(final_mat)

# def second_eq_term_2(Gamma_b,omega,N_b):
    
#     # Calculation is correct : Yash Palan 10/10/2023 12:38


#     """
#     Gamma_b: type - numply comlpesx array
#             size - 2N_b x 2N_b
#             This is the bosonic correlation matrix/ scattering matrix
#     delta_R: type - numpy complex array
#             size -1x2*N_b
#             This stores the expectatin values of all the Bosonic operators. 
    
#     omega: type - numpy array
#             this is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
#     N_b: type - float
#                     Number of bosons, or basically the number of momentum points in the k grid 
#                     (as the number of bosons is given by the discretisation of the bosonic annihilation 
#                     operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
#                     momentum grid).
#     -------------------------------------------------
#     Remarks
#     -------------------------------------------------
#     Checked: Works properly,  Yash Palan 10/10/2023 12:38
        
#     """  
#     #Making the matrix where the values get stored
#     final_mat = np.zeros([2*N_b, 2*N_b],dtype=complex)
    
#     Omega_mat_real = gf.Omega(N_b)
#     omega_bar_mat_real = gf.omega_bar(omega)
   
#     # Converting all the above matrices to complex types
#     Omega_mat = Omega_mat_real.astype(complex)
#     omega_bar_mat = omega_bar_mat_real.astype(complex)
    
#     for i in range(2*N_b):
#         for j in range(2*N_b):
#             temp=0
#             for ip in range(2*N_b):
#                 temp = temp + 2*(1j)*omega_bar_mat[ip]*( Omega_mat[i,ip]*Gamma_b[ip,j] + Omega_mat[j,ip]*Gamma_b[ip,i]   )
            
#             final_mat[i,j]=temp
    
#     return(final_mat)
    
def second_eq_term_2(Gamma_b,omega,N_b):
    
    # Calculation is correct : Yash Palan 10/10/2023 12:38


    """
    Gamma_b: type - numply comlpesx array
            size - 2N_b x 2N_b
            This is the bosonic correlation matrix/ scattering matrix
    delta_R: type - numpy complex array
            size -1x2*N_b
            This stores the expectatin values of all the Bosonic operators. 
    
    omega: type - numpy array
            this is the variation of omega(k), the frequency of the bosons (omega) with momentum (k).
    N_b: type - float
                    Number of bosons, or basically the number of momentum points in the k grid 
                    (as the number of bosons is given by the discretisation of the bosonic annihilation 
                    operator b(k) -> b_{i} where 1 <= i <= N. So, N_b = N which is basically the 
                    momentum grid).
                    
                    
    Return - final_mat : TYPE - numpy array
                         SIZE - 2N_b x 2N_b 
                         DESCRIPTION - This stores the 
                         
    -------------------------------------------------
    Remarks
    -------------------------------------------------
    
        
    """  
    #Making the matrix where the values get stored
    # final_mat = np.zeros([2*N_b, 2*N_b],dtype=complex)    
    Omega_mat = gf.Omega(N_b)
    omega_bar_mat = gf.omega_bar(omega,"complex")
    omega_bar_mat_transpose = omega_bar_mat.transpose()
    
    # Converting all the above matrices to complex types
    Omega_mat = Omega_mat.astype(complex)
    
    final_mat = -2*(1j)*(np.matmul(Gamma_b,omega_bar_mat.dot(Omega_mat)) 
                         + np.matmul(Gamma_b,omega_bar_mat_transpose.dot(Omega_mat))
                         - np.matmul(Omega_mat,omega_bar_mat.dot(Gamma_b))
                         - np.matmul(Omega_mat,omega_bar_mat_transpose.dot(Gamma_b))) 
    
    return(final_mat)    

##############################################################################
##############################################################################
# Third equation of motion functions
def third_eq_term_1(Gamma_b,Gamma_f,omega,N_b,N_f,J_0, delta_r, lmbda, momentum_array, position_array, volume):
    """

    Parameters
    ----------
    Gamma_b : type - numply comlpesx array
            size - 2N_b x 2N_b
            This is the bosonic correlation matrix/ scattering matrix
    Gamma_f : TYPE
        DESCRIPTION.
    omega : TYPE
        DESCRIPTION.
    N_b : TYPE
        DESCRIPTION.
    N_f : TYPE
        DESCRIPTION.
    J_0 : TYPE
        DESCRIPTION.
    delta_r :type - numpy complex array
            size -1x2*N_b
            This stores the expectatin values of all the Bosonic operators. 
    ip : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.
    lmbda : TYPE
        DESCRIPTION.
    momentum_array : TYPE
        DESCRIPTION.
    position_array : TYPE
        DESCRIPTION.
    volume : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    -------------------------------------------------
    Remarks
    -------------------------------------------------
    Checked: Works properly,  Yash Palan  9/10/2023 12:30
    

    """
    #Making the matrix where the values get stored
    final_mat = np.zeros([2*N_f, 2*N_f,2,2],dtype=complex)
    # for i in range(2*N_f):
    #     for j in range(2*N_f):
    #         for spin_alpha in range(2):
    #             for spin_beta in range(2):
    #                 val=0        
    #                 for ip in range(2*N_f):
    #                     for sigmap in range(2):
    #                         val = val + 4*1j*( -( (gf.J_i_j_prime(J_0, delta_r, Gamma_b, ip, i, N_b, N_f, lmbda, momentum_array, position_array, volume)) \
    #                                           - (gf.J_i_j_prime(J_0, delta_r, Gamma_b, i, ip, N_b, N_f, lmbda, momentum_array, position_array, volume)) )*Gamma_f[ip,j,spin_alpha,spin_beta]\
    #                                           + ( (gf.J_i_j_prime(J_0, delta_r, Gamma_b, ip, j, N_b, N_f, lmbda, momentum_array, position_array, volume)) \
    #                                                             - (gf.J_i_j_prime(J_0, delta_r, Gamma_b, j, ip, N_b, N_f, lmbda, momentum_array, position_array, volume)) )*Gamma_f[ip,i,spin_alpha,spin_beta] )
    #                 final_mat[i,j,spin_alpha,spin_beta] = val
    
    for i,j,spin_alpha,spin_beta in it.product(range(2*N_f),range(2*N_f),range(2),range(2)):
        val=0        
        for ip,sigmap in it.product(range(2*N_f),range(2)):
            val = val + 1j*( -( (gf.J_i_j_prime(J_0, delta_r, Gamma_b, ip, i, N_b, N_f, lmbda, momentum_array, position_array, volume)) \
                                - (gf.J_i_j_prime(J_0, delta_r, Gamma_b, i, ip, N_b, N_f, lmbda, momentum_array, position_array, volume)) )*Gamma_f[ip,j,sigmap,spin_beta]\
                            + ( (gf.J_i_j_prime(J_0, delta_r, Gamma_b, ip, j, N_b, N_f, lmbda, momentum_array, position_array, volume)) \
                                - (gf.J_i_j_prime(J_0, delta_r, Gamma_b, j, ip, N_b, N_f, lmbda, momentum_array, position_array, volume)) )*Gamma_f[ip,i,sigmap,spin_alpha] )
        final_mat[i,j,spin_alpha,spin_beta] = val
    return(final_mat)

def third_eq_term_2(delta_R,Gamma_b,Gamma_f,N_b,N_f,gamma,lmbda,omega,momentum_array,position_array,volume):
    """
    

    Parameters
    ----------
    Gamma_b : Type: complex numpy array
              Size : 2N_b x 2N_b 
              DESCRIPTION.
    Gamma_f : TYPE: complex numpy array
              Size : 2N_f x 2N_f x 2 x 2.
    N_b : Type : int
        DESCRIPTION.
    N_f : TYPE : int
        DESCRIPTION.
    gamma : TYPE: real numpy array
            Size : 2N_b x 1
        DESCRIPTION.
    lmbda : TYPE : real numpy array
            Size : 
        DESCRIPTION.
    omega : TYPE : 
            Size : 
        DESCRIPTION.
    momentum_array : TYPE : 
                     Size :
        DESCRIPTION.
    position_array : TYPE :
                     Size :
        DESCRIPTION.
    volume : TYPE :
             Size : 
        DESCRIPTION.

    Returns
    -------
    finaL_mat :  
        
    -------------------------------------------------
    Remarks
    -------------------------------------------------
    Checked: Works properly,  Yash Palan  9/10/2023 12:30

    """
    #Making the matrix where the values get stored
    final_mat = np.zeros([2*N_f, 2*N_f,2,2],dtype=complex)
    
    eta_mat = gf.eta(N_b,N_f,gamma,lmbda,omega,momentum_array,position_array,volume).astype("complex")
    
    for i,j,spin_alpha,spin_beta in it.product(range(2*N_f),range(2*N_f),range(2),range(2)):
        for l in range(2*N_b):
            final_mat[i,j,spin_alpha,spin_beta]+= -2*1j*delta_R[l]*(
                gf.get_value_1(eta_mat, l, i-N_f)*gf.get_value_2(Gamma_f, i-N_f, spin_alpha, j, spin_beta)
                +gf.get_value_1(eta_mat, l, j-N_f)*gf.get_value_2(Gamma_f, i, spin_alpha, j-N_f, spin_beta)
                -gf.get_value_1(eta_mat, l, i)*gf.get_value_2(Gamma_f, i+N_f, spin_alpha, j, spin_beta)
                -gf.get_value_1(eta_mat, l, j)*gf.get_value_2(Gamma_f, i, spin_alpha, j+N_f, spin_beta)   
                                            )    
    return(final_mat)

# def third_eq_term_3(Gamma_b,Gamma_f,N_b,N_f,gamma,lmbda,omega,momentum_array,position_array,volume):
#     """
    
#     Parameters
#     ----------
#     Gamma_b : TYPE
#         DESCRIPTION.
#     Gamma_f : TYPE
#         DESCRIPTION.
#     N_b : TYPE
#         DESCRIPTION.
#     N_f : TYPE
#         DESCRIPTION.
#     gamma : TYPE
#         DESCRIPTION.
#     lmbda : TYPE
#         DESCRIPTION.
#     omega : TYPE
#         DESCRIPTION.
#     momentum_array : TYPE
#         DESCRIPTION.
#     position_array : TYPE
#         DESCRIPTION.
#     volume : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     -------------------------------------------------
#     Remarks
#     -------------------------------------------------
#     Checked: Works properly,  Yash Palan 9/10/2023 12:30

#     """
    
#     #Making the matrix where the values get stored
#     final_mat = np.zeros([2*N_f, 2*N_f,2,2],dtype=complex)
    
#     # for i in range(2*N_f):
#     #     for j in range(2*N_f):
#     #         for spin_alpha in range(2):
#     #             for spin_beta in range(2):
                    
#     #                 val=0
#     #                 kappa_prime_mat_real=gf.kappa_prime(i,j,N_b,lmbda,omega,gamma,momentum_array,position_array,volume)
#     #                 kappa_prime_mat = kappa_prime_mat_real.astype("comlpex")
#     #                 for ip in range(N_f):
#     #                     for alphap in range(2):
                            
#     #                         # Note - we still need to check if the following equation is correct. It most likely is, but a
#     #                         # second check is never bad and hence it neeed to be checked (I mean the analytical expression)
#     #                         val += (1+Gamma_f[ip,ip+N_f,alphap,alphap])*( gf.get_value(kappa_prime_mat,ip,i-N_f)*gf.get_value_2(Gamma_f,i-N_f, spin_alpha,j,spin_beta) \
#     #                                                               +gf.get_value(kappa_prime_mat,ip,j-N_f)*gf.get_value_2(Gamma_f,i, spin_alpha,j-N_f,spin_beta) \
#     #                                                               -gf.get_value(kappa_prime_mat,ip,i)*gf.get_value_2(Gamma_f,i+N_f, spin_alpha,j,spin_beta) \
#     #                                                               -gf.get_value(kappa_prime_mat,ip,j)*gf.get_value_2(Gamma_f,i, spin_alpha,j+N_f,spin_beta) \
#     #                                                               +gf.get_value(kappa_prime_mat,i-N_f,ip)*gf.get_value_2(Gamma_f,i-N_f, spin_alpha,j,spin_beta) \
#     #                                                               +gf.get_value(kappa_prime_mat,j-N_f,ip)*gf.get_value_2(Gamma_f,i, spin_alpha,j-N_f,spin_beta) \
#     #                                                               -gf.get_value(kappa_prime_mat,i,ip)*gf.get_value_2(Gamma_f,i+N_f, spin_alpha,j,spin_beta) \
#     #                                                               -gf.get_value(kappa_prime_mat,j,ip)*gf.get_value_2(Gamma_f,i, spin_alpha,j+N_f,spin_beta) ) 
#     #                 final_mat[i,j,spin_alpha,spin_beta]=val
    
#     for i,j,spin_alpha,spin_beta in it.product(range(2*N_f),range(2*N_f),range(2),range(2)):
#         val=0
#         kappa_prime_mat=gf.kappa_prime(N_f,N_b,lmbda,omega,gamma,
#                                        momentum_array,position_array,volume).astype("complex")
#         for ip,alphap in it.product(range(N_f),range(2)):
#             # Note - we still need to check if the following equation is correct. It most likely is, but a
#             # second check is never bad and hence it neeed to be checked (I mean the analytical expression)
#             val += (1+Gamma_f[ip,ip+N_f,alphap,alphap])*( gf.get_value_1(kappa_prime_mat,ip,i-N_f)*gf.get_value_2(Gamma_f,i-N_f, spin_alpha,j,spin_beta) \
#                                                          +gf.get_value_1(kappa_prime_mat,ip,j-N_f)*gf.get_value_2(Gamma_f,i, spin_alpha,j-N_f,spin_beta) \
#                                                          -gf.get_value_1(kappa_prime_mat,ip,i)*gf.get_value_2(Gamma_f,i+N_f, spin_alpha,j,spin_beta) \
#                                                          -gf.get_value_1(kappa_prime_mat,ip,j)*gf.get_value_2(Gamma_f,i, spin_alpha,j+N_f,spin_beta) \
#                                                          +gf.get_value_1(kappa_prime_mat,i-N_f,ip)*gf.get_value_2(Gamma_f,i-N_f, spin_alpha,j,spin_beta) \
#                                                          +gf.get_value_1(kappa_prime_mat,j-N_f,ip)*gf.get_value_2(Gamma_f,i, spin_alpha,j-N_f,spin_beta) \
#                                                          -gf.get_value_1(kappa_prime_mat,i,ip)*gf.get_value_2(Gamma_f,i+N_f, spin_alpha,j,spin_beta) \
#                                                          -gf.get_value_1(kappa_prime_mat,j,ip)*gf.get_value_2(Gamma_f,i, spin_alpha,j+N_f,spin_beta) ) 
#             final_mat[i,j,spin_alpha,spin_beta]=val
    
#     return(final_mat)



def third_eq_term_3(Gamma_b,Gamma_f,N_b,N_f,gamma,lmbda,omega,momentum_array,position_array,volume):
    """
    
    Parameters
    ----------
    Gamma_b : TYPE
        DESCRIPTION.
    Gamma_f : TYPE
        DESCRIPTION.
    N_b : TYPE
        DESCRIPTION.
    N_f : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    lmbda : TYPE
        DESCRIPTION.
    omega : TYPE
        DESCRIPTION.
    momentum_array : TYPE
        DESCRIPTION.
    position_array : TYPE
        DESCRIPTION.
    volume : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    -------------------------------------------------
    Remarks
    -------------------------------------------------
    Checked: Works properly,  Yash Palan 9/10/2023 12:30

    """
    
    #Making the matrix where the values get stored
    final_mat = np.zeros([2*N_f, 2*N_f,2,2],dtype=complex)
    for i,j,spin_alpha,spin_beta in it.product(range(2*N_f),range(2*N_f),range(2),range(2)):
        val=0
        kappa_prime_mat=gf.kappa_prime(N_f,N_b,lmbda,omega,gamma,
                                       momentum_array,position_array,volume).astype("complex")
        for ip,jp,alphap,betap in it.product(range(N_f),range(N_f),range(2),range(2)):
            # Note - we still need to check if the following equation is correct. It most likely is, but a
            # second check is never bad and hence it neeed to be checked (I mean the analytical expression)
            val_1_1 = (-(1j)*kappa_prime_mat[ip,jp]*(gf.delta(ip, jp)*gf.delta(alphap,betap) -(1j)*Gamma_f[ip,jp,alphap,betap] ) 
                     *(gf.delta(i,jp+N_f)*gf.delta(spin_alpha,betap)*Gamma_f[ip+N_f,j,alphap,spin_beta] 
                       -gf.delta(ip+N_f,i)*gf.delta(alphap,spin_alpha)*Gamma_f[jp+N_f,j,betap,spin_beta]
                       +gf.delta(j,jp+N_f)*gf.delta(spin_beta,betap)*Gamma_f[i,ip+N_f,spin_alpha,alphap]
                       -gf.delta(ip+N_f,j)*gf.delta(alphap,spin_beta)*Gamma_f[i,jp+N_f,spin_alpha,betap]
                       ) 
                     )
            val_1_2 = (-(1j)*kappa_prime_mat[ip,jp]*(gf.delta(ip+N_f, jp+N_f)*gf.delta(alphap,betap) -(1j)*Gamma_f[ip+N_f,jp+N_f,alphap,betap] ) 
                     *(gf.delta(i,jp)*gf.delta(spin_alpha,betap)*Gamma_f[ip,j,alphap,spin_beta] 
                       -gf.delta(ip,i)*gf.delta(alphap,spin_alpha)*Gamma_f[jp,j,betap,spin_beta]
                       +gf.delta(j,jp)*gf.delta(spin_beta,betap)*Gamma_f[i,ip,spin_alpha,alphap]
                       -gf.delta(ip,j)*gf.delta(alphap,spin_beta)*Gamma_f[i,jp,spin_alpha,betap]
                       ) 
                     )
            
            kp = ip+N_f
            lp = jp+N_f
            val_3 = (-(1j)*(kappa_prime_mat[ip,jp]+kappa_prime_mat[jp,ip])*
                     (gf.delta(jp,kp)*gf.delta(betap,alphap)-(1j)*Gamma_f[jp,kp,betap,alphap])
                     *(gf.delta(i,lp)*gf.delta(spin_alpha,betap)*Gamma_f[ip,j,alphap,spin_beta]
                       -gf.delta(ip,i)*gf.delta(spin_alpha,alphap)*Gamma_f[lp,j,betap,spin_beta]
                       +gf.delta(j,lp)*gf.delta(spin_beta,betap)*Gamma_f[i,ip,spin_alpha,alphap]
                       -gf.delta(ip,j)*gf.delta(alphap,spin_beta)*Gamma_f[i,lp,spin_alpha,betap]
                         ))
            val_2 =  (-(1j)*(kappa_prime_mat[ip,jp]+kappa_prime_mat[jp,ip])*
                     (gf.delta(jp,kp)*gf.delta(betap,alphap)-(1j)*Gamma_f[jp,kp,betap,alphap])
                     *(gf.delta(i,lp)*gf.delta(spin_alpha,betap)*Gamma_f[ip,j,alphap,spin_beta]
                       -gf.delta(ip,i)*gf.delta(spin_alpha,alphap)*Gamma_f[lp,j,betap,spin_beta]
                       +gf.delta(j,lp)*gf.delta(spin_beta,betap)*Gamma_f[i,ip,spin_alpha,alphap]
                       -gf.delta(ip,j)*gf.delta(alphap,spin_beta)*Gamma_f[i,lp,spin_alpha,betap]
                         ))
            val = val_1_1+val_1_2+val_2+val_3
            final_mat[i,j,spin_alpha,spin_beta]=val
    
    return(final_mat)
##############################################################################
##############################################################################
    
