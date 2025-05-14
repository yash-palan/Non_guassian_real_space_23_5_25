# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:48:18 2023

@author: Yash_palan

This code contains the code that will
"""

#########################################
#########################################
import numpy as np
import torch
##############################################################################
##############################################################################
# Initialisation of the input matrices 

def initialise_delta_R_matrix(N_b:int,seed:float)->torch.Tensor:
    """
    -----------------------------
    -----------------------------
    DESCRIPTION
    -----------------------------
    -----------------------------
    Function to compute a random matrix used for the initialisation of the variational parameter Delta_R

    -----------------------------
    -----------------------------
    Parameters:
    -----------------------------
    -----------------------------
    seed:   Type: float
            Description: This is the seed used for the random number generator.

    N_b:    Type: int
            Description: This is the number of bosons in the system.
    -----------------------------
    -----------------------------
    Returns:
    -----------------------------
    -----------------------------
    Type: complex numpy array
    Shape: (2*N_b)     
    """
    # Set the seed
    # np.random.seed(seed)
    return(    torch.tensor(np.random.rand(2*N_b),dtype=torch.complex128)    ) 
