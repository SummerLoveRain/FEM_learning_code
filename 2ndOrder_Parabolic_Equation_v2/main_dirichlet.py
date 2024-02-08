import numpy as np
from fem import *
from solver import Parabolic_2ndOrder_FE_solver_Dirichlet
from basis import Basis

def u_func(x, y, t, derivative_order_x, derivative_order_y):
    '''
    解析解
    '''
    if derivative_order_x==0 and derivative_order_y==0:
        u = np.exp(x + y + t)
    elif derivative_order_x==1 and derivative_order_y==0:
        u = np.exp(x + y + t)
    elif derivative_order_x==0 and derivative_order_y==1:
        u = np.exp(x + y + t)
    else:
        ValueError("The derivative order is not correct!")
    return u

def coefficient_function(x, y):
    coe = 2
    return coe

def source_function(x, y, t):
    sour = -3*np.exp(x+y+t)
    return sour

if __name__ == '__main__':

    for i in range(5):
        bounds = [0, 2, 0, 1, 0, 1]
        Nx = 8*(2**i)
        Ny = 4*(2**i)
        Nt = 4*(2**i)
        theta = 1/2
        solver = Parabolic_2ndOrder_FE_solver_Dirichlet(bounds, Nx, Ny, Nt, theta, coefficient_function=coefficient_function, source_function=source_function, \
                                                basis_function_type=Basis.LINEAR)

        # 打印误差
        log_str = 'Nx %5d Ny %5d Nt %5d L_infinity: %10.6e L2: %10.6e H1: %10.6e H1-semi: %10.6e' % (Nx, Ny, Nt, solver.L_infinity_error(u_func), solver.L2_error(u_func), \
                                                                                solver.H1_error(u_func), solver.H1_semi_error(u_func))
        print(log_str)

    print("\n")

        
