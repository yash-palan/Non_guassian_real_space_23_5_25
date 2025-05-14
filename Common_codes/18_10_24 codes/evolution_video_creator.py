# -*- coding: utf-8 -*-
"""
Created on Oct 1 2024

@author: Yash_palan

This code contains the code that will create time evolution videos
"""

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
from functools import partial
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from Common_codes import correlation_functions_file_18_10_24 as cff

def extraction_of_diagonal_of_c_dagger_c_data(Gamma_m_complete:np.ndarray,N_f)->np.ndarray:
    c_dagger_c_data = np.zeros((np.shape(Gamma_m_complete)[0],int(N_f/2)))
    final_array = np.zeros((np.shape(Gamma_m_complete)[0],int(N_f/2)))
    for i in range(np.shape(Gamma_m_complete)[0]):
        Gamma_m = np.reshape(Gamma_m_complete[i,:],(2*N_f,2*N_f))
        c_dagger_c_data = cff.c_dagger_c_expectation_value_matrix_creation(Gamma_m,N_f)
        final_array[i,:] += np.real(np.diag(c_dagger_c_data)[0:int(N_f/2)])

    return(final_array)

def function_to_create_1D_video(data:np.ndarray,filename)->None:
    """
    Parameters
    ----------
    data : np.array
        2D array of data to be plotted.
        Size: time x data
    filename : str
        Name of the file to save the video.

    """
    # Step 1: Set up the figure, axis, and initial pcolormesh plot
    fig, ax = plt.subplots()

    # Initial pcolormesh plot
    line, = ax.plot(data[0, :])

    # Step 2: Define the update function for each frame
    def update(frame):
        line.set_ydata(data[frame, :])  # Update the mesh data for the current frame
        ax.set_title(f"Time Step {frame}")
        ax.set_ylim(min(data[frame, :]), max(data[frame, :]))
        return line,

    # Step 3: Create the animation
    ani = FuncAnimation(fig, update,frames = data.shape[0])

    # Step 4: Save or display the animation
    ani.save(filename+'.mp4', writer='ffmpeg', fps=10)  
    return    
  
def function_to_create_2D_video(data,square_size:int,filename):
    # Step 1: Set up the figure, axis, and initial pcolormesh plot
    fig, ax = plt.subplots()

    # Initial pcolormesh plot
    mesh = ax.pcolormesh(np.reshape(data[0, :],(square_size,square_size)), shading='auto', cmap='viridis')

    norm = Normalize(vmin=np.min(data[0,:]), vmax=np.max(data[0,:]))
    # Add a color bar to show the data range
    cbar = plt.colorbar(mesh, ax=ax, norm = norm)

    # Step 2: Define the update function for each frame
    def update(frame):
        mesh.set_array(data[frame, :])  # Update the mesh data for the current frame
        ax.set_title(f"Time Step {frame}")
        norm = Normalize(vmin=np.min(data[frame,:]), vmax=np.max(data[frame,:]))
        mesh.set_norm(norm)
        cbar.update_normal(mesh)
        return mesh,

    # Step 3: Create the animation
    ani = FuncAnimation(fig, update,frames = data.shape[0])

    # Step 4: Save or display the animation
    ani.save(filename+'.mp4', writer='ffmpeg', fps=10) 
    return       


if __name__ == "__main__":
    
    filename = "imag_time_evo_delta_r_complete.npy"
    data = np.load(filename)
    square_size = 10 
    # video_filename = "Delta_r_x_expectation_value_momentum_space_evolution"
    # function_to_create_1D_video(data[:,0:100],square_size,video_filename)
    # video_filename = "Delta_r_p_expectation_value_momentum_space_evolution"
    # function_to_create_1D_video(data[:,100:],square_size,video_filename)

    video_filename = "Delta_r_momentum_space_evolution"
    function_to_create_1D_video(data,video_filename)

    filename = "imag_time_evo_gamma_b_complete.npy"
    data = np.load(filename)

    filename = "imag_time_evo_gamma_m_complete.npy"
    data = np.load(filename)
    
    