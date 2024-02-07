import numpy as np
from fem import *
from solver import Possion_2D_FE_solver_Dirichlet
from basis import Basis

def u_func(x, y, derivative_order_x, derivative_order_y):
    '''
    解析解
    '''
    if derivative_order_x==0 and derivative_order_y==0:
        u = x*y*(1-x/2)*(1-y)*np.exp(x+y)
    elif derivative_order_x==1 and derivative_order_y==0:
        u = (y-y**2-x**2*y/2+x**2*y**2/2)*np.exp(x+y)
    elif derivative_order_x==0 and derivative_order_y==1:
        u = (x-x*y-x*y**2-x**2/2+x**2*y/2+x**2*y**2/2)*np.exp(x+y)
    else:
        ValueError("The derivative order is not correct!")
    return u

def coefficient_function(x, y):
    coe = 1
    return coe

def source_function(x, y):
    sour = -y*(1-y)*(1-x-x**2/2)*np.exp(x+y)-x*(1-x/2)*(-3*y-y**2)*np.exp(x+y)
    return sour

if __name__ == '__main__':

    for i in range(5):
        bounds = [-1, 1, -1, 1]
        Nx = 8*(2**i)
        Ny = 8*(2**i)
        solver = Possion_2D_FE_solver_Dirichlet(bounds, Nx, Ny, coefficient_function=coefficient_function, source_function=source_function, \
                                                basis_function_type=Basis.QUADRATIC)

        # 打印误差        
        log_str = 'Nx %5d Ny %5d L_infinity: %10.6e L2: %10.6e H1: %10.6e H1-semi: %10.6e' % (Nx, Ny, solver.L_infinity_error(u_func), solver.L2_error(u_func), \
                                                                                solver.H1_error(u_func), solver.H1_semi_error(u_func))
        print(log_str)

    print("\n")

        
