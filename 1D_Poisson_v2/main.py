import numpy as np
from fem import *
from solver import Possion_1D_FE_solver_Dirichlet, Possion_1D_FE_solver_Neumann, Possion_1D_FE_solver_Robin
from basis import Basis

def Exact_u(x):
    '''
    解析解
    '''
    u = x * np.cos(x)
    return u

def Exact_u_derivative(x):
    '''
    解析解
    '''
    u = np.cos(x) - x*np.sin(x)
    return u

def coefficient_function(x):
    coe = np.exp(x)
    return coe

# def boundary_function(a, b, x):
#     if np.abs(x-a)<1e-15:
#         result = 0.0
#     elif np.abs(x-b)<1e-15:
#         result = np.cos(b)
#     else:
#         ValueError('No boundary points!')
#     return result

def source_function(x):
    sour = -np.exp(x)*(np.cos(x)-2.0*np.sin(x)-x*np.cos(x)-x*np.sin(x))
    return sour

if __name__ == '__main__':
    a = 0.0
    b = 1.0
    Gaussian_Integral_1D_N = 3
    Nh = 6
    
    for i in range(Nh):
        number_of_elements = 2**(i+2)
        solver = Possion_1D_FE_solver_Dirichlet(a, b, coefficient_function, source_function, number_of_elements=number_of_elements, \
                                      basis_function_type_trial=Basis.LINEAR, basis_function_type_test=Basis.LINEAR, Gaussian_Integral_1D_N=Gaussian_Integral_1D_N)
        # 打印误差        
        log_str = 'n %5d L_infinity: %10.6e L2: %10.6e H1: %10.6e H1-semi: %10.6e' % (number_of_elements, solver.L_infinity_error(Exact_u), solver.L2_error(Exact_u), \
                                                                                solver.H1_error(Exact_u, Exact_u_derivative), solver.H1_semi_error(Exact_u_derivative))
        print(log_str)

    print("\n")

        
    for i in range(Nh):
        number_of_elements = 2**(i+2)
        solver = Possion_1D_FE_solver_Neumann(a, b, coefficient_function, source_function, number_of_elements=number_of_elements, \
                                      basis_function_type_trial=Basis.QUADRATIC, basis_function_type_test=Basis.QUADRATIC, Gaussian_Integral_1D_N=Gaussian_Integral_1D_N)

        # 打印误差        
        log_str = 'n %5d L_infinity: %10.6e L2: %10.6e H1: %10.6e H1-semi: %10.6e' % (number_of_elements, solver.L_infinity_error(Exact_u), solver.L2_error(Exact_u), \
                                                                                solver.H1_error(Exact_u, Exact_u_derivative), solver.H1_semi_error(Exact_u_derivative))
        print(log_str)

    print("\n")
    
        
    for i in range(Nh):
        number_of_elements = 2**(i+2)
        solver = Possion_1D_FE_solver_Robin(a, b, coefficient_function, source_function, number_of_elements=number_of_elements, \
                                      basis_function_type_trial=Basis.QUADRATIC, basis_function_type_test=Basis.QUADRATIC, Gaussian_Integral_1D_N=Gaussian_Integral_1D_N)

        # 打印误差  
        log_str = 'n %5d L_infinity: %10.6e L2: %10.6e H1: %10.6e H1-semi: %10.6e' % (number_of_elements, solver.L_infinity_error(Exact_u), solver.L2_error(Exact_u), \
                                                                                solver.H1_error(Exact_u, Exact_u_derivative), solver.H1_semi_error(Exact_u_derivative))
        print(log_str)
