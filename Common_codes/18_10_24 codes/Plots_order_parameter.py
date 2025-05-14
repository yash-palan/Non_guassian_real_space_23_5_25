import numpy as np
import math as math 
import matplotlib.pyplot as plt
from logging import raiseExceptions
from Common_codes import global_functions_v3_pytorch_implement_24_7_24 as gf
# from Common_codes import global_function_old as gf
import os 
# from Common_codes import global_function_github as gf  

# this module takes the calculated Delta_r, Gamma_b and Gamma_m matrices and plots the order parameters as required.

number_of_points = 10
positon_value_max = [10 , 10]
positon_value_min = [0  , 0]
position_space_grid = gf.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points,True)
print("Position space grid created")

momentum_value_max = [np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
momentum_value_min = [-np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points ,-np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points]
momentum_space_grid = gf.coordinate_array_creator_function(momentum_value_min,momentum_value_max,number_of_points, False)
print("Momentum space grid created")

volume = np.prod(np.array(positon_value_max)-np.array(positon_value_min))

k_x = np.linspace(-np.pi, np.pi, number_of_points)
k_y = np.linspace(-np.pi, np.pi, number_of_points)

x= np.linspace(0, 10, number_of_points)
y= np.linspace(0, 10, number_of_points)

def fourier_transform_delta_r(delta_r:np.ndarray, position_array:np.ndarray, momentum_array:np.ndarray, volume:float, spin_index)->np.ndarray:
    """
    This function just does the Fourier transform of the delta_r array (since we treat bosons in the Fourier space, while
    we treat the Fermions in the position space). Also, the relationship 
    """

    plt.plot(delta_r)
    plt.show()

    delta_x_q = delta_r[0:int(delta_r.shape[0]/2)]
    delta_p_q = delta_r[int(delta_r.shape[0]/2):]

    if spin_index == True:  
        position_array_new = position_array[0:int(position_array.shape[0]/2),:]

    expectation_value_of_b_k = 1/2*(delta_x_q+1j*delta_p_q) 

    if(delta_x_q.shape[0]!=delta_p_q.shape[0]):
        raiseExceptions("The size of the delta_x and delta_p arrays are not the same. Check the breakup again.")
    
    if(position_array.ndim==1):
        matrix = np.einsum('i,k->ki', position_array_new, momentum_array)
    if(position_array.ndim>1):
        matrix = np.einsum('id,kd->ki', position_array_new, momentum_array)
    
    expectation_value_of_b_i =(1/np.sqrt(volume))*np.einsum('k,ki->i',expectation_value_of_b_k, np.exp(1j*matrix))
    
    delta_x_position_space = 2*np.real(expectation_value_of_b_i)
    delta_p_position_space = 1j*(np.conjugate( expectation_value_of_b_i) - expectation_value_of_b_i)

    return delta_x_position_space, delta_p_position_space

def plot_delta(delta_r: np.ndarray, position_array:np.ndarray, momentum_arry:np.ndarray, volume:float, spin_index = True)->None:
    """
    This function plots the delta_r in the position space and momentum space.
    """
    delta_x_position_space, delta_p_position_space = fourier_transform_delta_r(delta_r, position_array, momentum_arry, volume, spin_index)

    if spin_index == True:
        position_array_spinless = position_array[0:int(position_array.shape[0]/2),:]
    
    linear_dimension = int(math.sqrt(int(position_array_spinless.shape[0])))
    print("linear_dimension = ", linear_dimension)

    delta_x_position_space_grid = delta_x_position_space.reshape(( linear_dimension, linear_dimension))
    delta_p_position_space_grid = delta_p_position_space.reshape(( linear_dimension, linear_dimension))

    plt.pcolormesh(delta_x_position_space_grid)
    plt.colorbar()
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$y$', fontsize=20)
    plt.title(r'$ \langle b^{\dagger}_{i} + b_{i} \rangle$ ', fontsize=20)
    # plt.savefig(figure_path + f"Delta_r_{mu}.png")
    # plt.savefig(figure_path + "1.png")
    plt.show()

    print("maximum imaginary part in delta_x is ", max(delta_x_position_space.imag))

    plt.pcolormesh(delta_p_position_space_grid.real)
    plt.colorbar()
    plt.xlabel(r'$k_x$', fontsize=20)
    plt.ylabel(r'$k_y$', fontsize=20)
    plt.title(r'$ \text{Displacement} (\Delta_p)$ in position space')
    # plt.savefig(figure_path + f"delta_p_{mu}.png")
    # plt.savefig(figure_path + "2.png")
    plt.show()

    print("maximum imaginary part in delta_p is ", max(delta_p_position_space.imag))
    return None


def plot_gamma_b(gamma_b: np.ndarray, position_array: np.ndarray, momentum_array: np.ndarray, volume: float, spin_index = True):

    plt.pcolormesh(gamma_b.real)
    plt.colorbar()
    plt.xlabel(r'$q_1$', fontsize=20)
    plt.ylabel(r'$q_2$', fontsize=20)
    plt.title(r'$\Gamma_b $' , fontsize=20)
    # plt.savefig(figure_path + "Gamma_b.png")
    # plt.savefig(figure_path + "17.png")
    plt.show()

    N_b = int(gamma_b.shape[0])
    if spin_index == True:
        position_array_spinless = N_b/2
    
    linear_dimension = int(math.sqrt(position_array_spinless))  

    gamma_b_diag = np.diag(gamma_b[0:int(N_b/2)])

    gamma_b_grid = gamma_b_diag.reshape((linear_dimension, linear_dimension))
    
    plt.contourf(k_x, k_y, gamma_b_grid.real, levels = 40)
    plt.colorbar()
    plt.xlabel(r'$q_x$', fontsize=20)
    plt.ylabel(r'$q_y$', fontsize=20)
    plt.title(r'$\Gamma_b $' , fontsize=20)
    # plt.savefig(figure_path + f"Gamma_b_{mu}.png") 
    # plt.savefig(figure_path + "18.png")
    plt.show()

def plot_fermionic_correlations(Gamma_m: np.ndarray, position_array: np.ndarray, momentum_array: np.ndarray, volume: float, spin_index = True)->np.ndarray:

    print("Plotting the fermionic correlations in the majorana basis")
    plt.pcolormesh(Gamma_m.real)
    plt.colorbar()
    plt.xlabel(r'$i, \sigma_1$', fontsize=20)
    plt.ylabel(r'$j, \sigma_2$', fontsize=20)
    plt.title(r'$\Gamma_m$' , fontsize=20)
    # plt.savefig(figure_path + f"Gamma_m_{mu}.png")
    # plt.savefig(figure_path + "3.png")
    plt.show()


    N_f = int(position_array.shape[0])
    c_dagger_c_matrix = np.real(gf.c_dagger_c_expectation_value_matrix_creation(Gamma_m, N_f))
    # c_dagger_c_matrix = np.real(gf.c_dagger_c_expectation_value_matrix_creation_2(Gamma_m, N_f))
    
    if spin_index == True:  
        N_f_spinless = int(N_f/2)

    linear_dimension = int(math.sqrt(N_f_spinless))

    print("Plotting the number operator in real space")
    plt.pcolormesh(c_dagger_c_matrix.real)
    plt.colorbar()
    plt.title(r'$ \langle c^{\dagger}_{i \sigma_1} c_{j \sigma_2} \rangle $', fontsize = 20)
    plt.xlabel(r'$i \sigma_1$', fontsize=20)
    plt.ylabel(r'$j \sigma_2$', fontsize=20)
    # plt.savefig(figure_path + "c_dagger_c_matrix.png")
    # plt.savefig(figure_path + "4.png")
    plt.show()

    number_operator_spin_up_real_space =  c_dagger_c_matrix[0: N_f_spinless, 0:N_f_spinless]
    number_operator_spin_down_real_space =  c_dagger_c_matrix[N_f_spinless:, N_f_spinless:]

    plt.pcolormesh(number_operator_spin_up_real_space.real)
    plt.colorbar()
    plt.xlabel(r'$i \uparrow$', fontsize=20)
    plt.ylabel(r'$j \uparrow$', fontsize=20)
    plt.title(r"$ \langle c^{\dagger}_{i \uparrow} c_{j \uparrow} \rangle $", fontsize=20)
    # plt.savefig(figure_path + "number_operator_spin_up_real_space.png")
    # plt.savefig(figure_path + "5.png")
    plt.show()


    plt.pcolormesh(number_operator_spin_down_real_space.real)
    plt.colorbar()
    plt.xlabel(r'$i \downarrow$', fontsize=20)
    plt.ylabel(r'$j \downarrow$', fontsize=20)
    plt.title(r"$ \langle c^{\dagger}_{i \downarrow} c_{j \downarrow} \rangle $", fontsize=20)
    # plt.savefig(figure_path + "number_operator_spin_down_real_space.png")
    # plt.savefig(figure_path + "6.png")
    plt.show()

    print("plotting the total density on a x, y grid")
    total_number_operator_real_space = number_operator_spin_up_real_space + number_operator_spin_down_real_space
    diagonal_number_operator = np.diag(total_number_operator_real_space)

    print("filling fraction is ", np.trace(total_number_operator_real_space)/N_f)

    number_operator_grid = (diagonal_number_operator).reshape((linear_dimension, linear_dimension))
    plt.pcolormesh(number_operator_grid.real)
    plt.colorbar()
    plt.xlabel(r'$i$', fontsize=20)
    plt.ylabel(r'$j$', fontsize=20)
    plt.title(r'$ \sum_{\sigma} \langle c^{\dagger}_{i \sigma} c_{i \sigma} \rangle $', fontsize=20)
    # plt.savefig(figure_path + "total_number.png")
    # plt.savefig(figure_path + "7.png")
    plt.show()

    number_operator_diagonal_spin_up =  np.diag(number_operator_spin_up_real_space)
    number_operator_diagonal_spin_down =  np.diag(number_operator_spin_down_real_space)

    number_operator_diagonal_spin_up_grid = number_operator_diagonal_spin_up.reshape((linear_dimension, linear_dimension))
    number_operator_diagonal_spin_down_grid = number_operator_diagonal_spin_down.reshape((linear_dimension, linear_dimension))




    plt.pcolormesh(number_operator_diagonal_spin_up_grid.real)
    plt.colorbar()
    plt.xlabel(r'$i$', fontsize=20)
    plt.ylabel(r'$j$', fontsize=20)
    plt.title(r'$ \langle c^{\dagger}_{i \uparrow} c_{i \uparrow} \rangle $', fontsize=20)
    # plt.savefig(figure_path + f"number_operator_diagonal_{mu}.png")
    # plt.savefig(figure_path + "8.png")
    plt.show()

    plt.pcolormesh(number_operator_diagonal_spin_down_grid.real)
    plt.colorbar()
    plt.xlabel(r'$i$', fontsize=20)
    plt.ylabel(r'$j$', fontsize=20)
    plt.title(r'$ \langle c^{\dagger}_{i \downarrow} c_{i \downarrow} \rangle $', fontsize=20)
    # plt.savefig(figure_path + "number_operator_diagonal_spin_down_real_space.png")
    # plt.savefig(figure_path + "9.png")
    plt.show()

    fourier_tranform_number_operator = np.fft.fft2(number_operator_diagonal_spin_up_grid)
    plt.pcolormesh(np.abs(fourier_tranform_number_operator))
    plt.colorbar()
    plt.title("Fourier transform of the number operator numpy")
    plt.show()


    print("performing the fourier transform of the number operator")

    if spin_index == True: 
        position_array_spinless = position_array[0:int(position_array.shape[0]/2),:]

    if(position_array.ndim==1):
        matrix = np.einsum('i,k->ki',position_array_spinless, momentum_array)
    if(position_array.ndim>1):
        matrix = np.einsum('id,kd->ki',position_array_spinless, momentum_array)
    
    ft_matrix = np.exp(1j*matrix) #fourier transform matrix
    ift_matrix = np.exp(-1j*matrix)    #inverse fourier transform matrix

    number_operator_spin_up_momentum_space = 1/(volume)*np.einsum('qi, ij, kj-> qk', ft_matrix, number_operator_spin_up_real_space, ift_matrix)
    number_operator_spin_down_momentum_space = 1/(volume)*np.einsum('qi, ij, kj-> qk', ft_matrix, number_operator_spin_down_real_space, ift_matrix)

    staggered_momentum_space = 1/volume*np.einsum('qi, ij, kj-> qk', ft_matrix, total_number_operator_real_space, ift_matrix)

    number_operator_spin_up_momentum_space_grid = (number_operator_spin_up_momentum_space.real)
    number_operator_spin_down_momentum_space_grid = (number_operator_spin_down_momentum_space.real)
    
    plt.pcolormesh(number_operator_spin_up_momentum_space_grid)
    plt.colorbar()
    plt.xlabel(r'$k_{1}$', fontsize=20)
    plt.ylabel(r'$k_{2}$', fontsize=20)
    plt.title(r'$ \langle c^{\dagger}_{k_{1} \uparrow} c_{k_{2} \uparrow} \rangle $', fontsize=20)
    # plt.savefig(figure_path + "number_operator_spin_up_momentum_space.png")
    # plt.savefig(figure_path + "8.png")
    plt.show()

    plt.pcolormesh(number_operator_spin_down_momentum_space_grid)
    plt.colorbar()
    plt.xlabel(r'$k_1$', fontsize=20)
    plt.ylabel(r'$k_2$', fontsize=20)
    plt.title(r'$ \langle c^{\dagger}_{k_{1} \downarrow} c_{k_{2} \downarrow} \rangle $', fontsize=20)
    # plt.savefig(figure_path + "number_operator_spin_down_momentum_space.png")
    # plt.savefig(figure_path + "9.png")
    plt.show()

    number_operator_total_momentum_space = number_operator_spin_up_momentum_space_grid + number_operator_spin_down_momentum_space_grid
    plt.pcolormesh(number_operator_total_momentum_space.real)
    plt.colorbar()
    plt.title(r'$ \sum_{\sigma} \langle c^{\dagger}_{k_{1} \sigma} c_{k_{2} \sigma} \rangle $', fontsize=20)
    plt.xlabel(r'$k_{1}$', fontsize=20)
    plt.ylabel(r'$k_{2}$', fontsize=20)
    # plt.savefig(figure_path + "number_operator_total_momentum_space.png")
    # plt.savefig(figure_path + "10.png")
    plt.show()



    print("plotting the superconducting order parameter in real space")
    c_c_matrix = (gf.c_c_expectation_value_matrix_creation(Gamma_m, N_f))
    
    plt.pcolormesh(c_c_matrix.real)
    plt.colorbar()
    plt.xlabel(r'$i$', fontsize=20)
    plt.ylabel(r'$j$', fontsize=20)
    plt.title(r'$ \langle c_{i \sigma_1} c_{j \sigma_2} \rangle  $', fontsize=20)
    # plt.savefig(figure_path + "c_c_matrix.png")
    # plt.savefig(figure_path + "11.png")
    plt.show()

    plt.pcolormesh(c_c_matrix.imag)
    plt.colorbar()
    plt.xlabel(r'$i$', fontsize=20)
    plt.ylabel(r'$j$', fontsize=20)
    plt.title(r'$ \langle c_{i \sigma_1} c_{j \sigma_2} \rangle  $ imaginary', fontsize=20)
    # plt.savefig(figure_path + "supercoducting_pairing_real_space_53.png")
    plt.show()


    sop_real_space_1 =  c_c_matrix[N_f_spinless:, 0:N_f_spinless]
    sop_real_space_2 =  c_c_matrix[0:N_f_spinless, N_f_spinless:]

    plt.pcolormesh(sop_real_space_1.real)
    plt.colorbar()
    plt.title(r'$ \langle c_{i \uparrow} c_{j \downarrow} \rangle $', fontsize=20)
    plt.xlabel(r'$i \uparrow$', fontsize=20)
    plt.ylabel(r'$j \downarrow$', fontsize=20)
    # plt.savefig(figure_path + "superconducting_pairing_realpart_53.png")
    # plt.savefig(figure_path + "12.png")
    plt.show()

    plt.pcolormesh(sop_real_space_1.imag)
    plt.colorbar()
    plt.title(r'$ \langle c_{i \uparrow} c_{j \downarrow} \rangle $ imaginary', fontsize=20)
    plt.xlabel(r"$i \uparrow$", fontsize=20)
    plt.ylabel(r"$j \downarrow$", fontsize=20)
    # plt.savefig(figure_path + "superconducting_pairing_imaginary_53.png")
    plt.show()


    plt.pcolormesh(sop_real_space_2.real)
    plt.colorbar()
    plt.title(r'$ \langle c_{i \downarrow} c_{j \uparrow} \rangle $', fontsize=20)
    plt.xlabel(r'$i \downarrow$', fontsize=20)
    plt.ylabel(r'$j \uparrow$', fontsize=20)
    # plt.savefig(figure_path + "sop_real_space_2.png")
    # plt.savefig(figure_path + "13.png")
    plt.show()

    plt.pcolormesh(sop_real_space_2.imag)
    plt.colorbar()
    plt.title(r'$ \langle c_{i \downarrow} c_{j \uparrow} \rangle $ imaginary', fontsize=20)
    # plt.savefig(figure_path + "sop_real_space_53_imaginary.pdf")
    plt.show()


    print("performing the fourier transform of the superconducting order parameter")

# Yash 2/10/2024
# Q. I am Not sure why we have the ift_matrix here since the SC order parameter is actually <c_{k} c_{l}>, which would have the same sign?
#  In addition, I am not sure why he is getting correct results then? Because the equation is iteself incorrect here.
    superconducting_order_parameter_momentum_space_1 = np.einsum('qi, ij, kj-> qk', ft_matrix, sop_real_space_1, ift_matrix)/volume
    superconducting_order_parameter_momentum_space_2 = np.einsum('qi, ij, kj-> qk', ft_matrix, sop_real_space_2, ift_matrix)/volume



    superconducting_order_parameter_momentum_space_grid_1 = superconducting_order_parameter_momentum_space_1
    superconducting_order_parameter_momentum_space_grid_2 = superconducting_order_parameter_momentum_space_2

    print("plotting the superconducting order parameter in momentum space")
    plt.pcolormesh(np.real(superconducting_order_parameter_momentum_space_grid_1))
    plt.colorbar()
    plt.xlabel(r'$p_1$', fontsize=20)
    plt.ylabel(r'$p_2$', fontsize=20)
    plt.title(r'$ \langle c_{-p_{1}  \uparrow}c_{ p_{2} \downarrow} \rangle $', fontsize=20)
    # plt.savefig(figure_path + "superconducting_order_parameter_momentum_space_grid_1.png")
    # plt.savefig(figure_path + "14.png")
    plt.show()

    plt.pcolormesh(np.real(superconducting_order_parameter_momentum_space_grid_2))
    plt.colorbar()
    plt.xlabel(r'$p_1$', fontsize=20)
    plt.ylabel(r'$p_2$', fontsize=20)
    plt.title(r'$ \langle c_{-p_{1}  \downarrow}c_{p_{2} \uparrow} \rangle $', fontsize=20)
    # plt.savefig(figure_path + "supeconducting_pairing_momentum_space_53.pdf")
    # plt.savefig(figure_path + "15.png")
    plt.show()

    print("plotting the diagonal part of the superconducting order parameter in momentum space")
    diagonal_sop_momentum_space = np.diag(superconducting_order_parameter_momentum_space_grid_1 ) 
    print("The size of diagonal part of the superconducting order parameter in momentum space is ", np.shape(diagonal_sop_momentum_space))
    diagonal_sop_momentum_space_grid = diagonal_sop_momentum_space.reshape((linear_dimension, linear_dimension))

    print("overage_supercondcting_gap=", np.mean(np.abs(diagonal_sop_momentum_space_grid)))
    print("maximum_superconducting_gap=", np.max(np.abs(diagonal_sop_momentum_space_grid)))

    plt.contourf(k_x, k_y, np.abs(diagonal_sop_momentum_space_grid), levels = 10)
    plt.colorbar()
    plt.xlabel(r'$p_x$', fontsize=20)
    plt.ylabel(r'$p_y$', fontsize=20)
    plt.title(r'$\langle c_{-p \uparrow} c_{p_ \downarrow} \rangle $', fontsize=20)
    # plt.savefig(figure_path + f"abs_sop_{mu}.png")
    # plt.savefig(figure_path + "Superconducting_pairing_53_abs.pdf")
    plt.show()

    plt.contourf(k_x, k_y, np.angle(diagonal_sop_momentum_space_grid))
    plt.colorbar()
    plt.xlabel(r'$p_x$', fontsize=20)
    plt.ylabel(r'$p_y$', fontsize=20)
    plt.title(r'$ \langle c_{-p \uparrow} c_{p_ \downarrow} \rangle $', fontsize=20)
    # plt.savefig(figure_path + "diagonal_sop_momentum_space_contour.png")
    # plt.savefig(figure_path + f"phase_sop_{mu}.png")
    plt.show()

    return None



# path = "/home/vksharma/Documents/Variational_Method_Codes_2024/correct_data/"
# path = "/home/vksharma/Documents/Variational_Method_Codes_2024/data_sept_15_24/"
current_directory = os.getcwd() 
print("Current directory is", current_directory)

path  = current_directory 

figure_path = "/home/vksharma/Documents/Variational_Method_Codes_2024/data_sept_15_24/figures/"
figure_path = "/Users/vishal/Documents/nongaussian_variational_method/data/"

mu = 50
print("Path to the data is ", path) 
delta_r = np.load(path + "/Delta_R_final_50.npy")
Gamma_m = np.load(path + "/Gamma_m_final_50.npy") 
Gamma_b = np.load(path + "/Gamma_b_final_50.npy")

# print(np.multiply(Gamma_m, Gamma_m))

plot_delta(delta_r, position_space_grid, momentum_space_grid, volume, True)
plot_fermionic_correlations(Gamma_m, position_space_grid, momentum_space_grid, volume, True)
plot_gamma_b(Gamma_b, position_space_grid, momentum_space_grid, volume, True)


# list_mu = [-6.0, -5.7, -5.3. -5.0, -4.7, -4.4, -4.0]

# list_mu = [-4.4, -4.7, -5.3, -5.0, ] 

# list_filling_fraction = [0.7125335407600362, 0.5546181919036012, 0.3368680319725131, 0.49, ]


# sop = [0.15766974398611605, 0.20806466534205798, 0.34381944999451597, 0, ]

# max_sop = [0.22889795179800476, 0.2582675891575865, 0.46347679958749977, 0,  ]
