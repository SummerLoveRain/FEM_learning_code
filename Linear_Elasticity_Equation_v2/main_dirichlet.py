import numpy as np
from fem import *
from solver import Elasticity_2D_FE_solver_Dirichlet
from basis import Basis

def u1_func(x, y, derivative_order_x, derivative_order_y):
    '''
    解析解
    '''
    if derivative_order_x==0 and derivative_order_y==0:
        u = np.sin(np.pi*x)*np.sin(np.pi*y)
    elif derivative_order_x==1 and derivative_order_y==0:
        u = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
    elif derivative_order_x==0 and derivative_order_y==1:
        u = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
    else:
        ValueError("The derivative order is not correct!")
    return u

def u2_func(x, y, derivative_order_x, derivative_order_y):
    '''
    解析解
    '''
    if derivative_order_x==0 and derivative_order_y==0:
        u = x*(x-1)*y*(y-1)
    elif derivative_order_x==1 and derivative_order_y==0:
        u = 2*x*y**2 - 2*x*y - y**2 + y
    elif derivative_order_x==0 and derivative_order_y==1:
        u = 2*x**2*y - x**2 -2*x*y + x
    else:
        ValueError("The derivative order is not correct!")
    return u

def source1_function(x, y, lambda_, mu):
    sour = -(lambda_+2*mu)*(-np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)) - \
            (lambda_+mu)*((2*x-1)*(2*y-1)) - mu*(-np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y))
    return sour

def source2_function(x, y, lambda_, mu):
    sour = -(lambda_+2*mu)*(2*x*(x-1)) - \
            (lambda_+mu)*(np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y)) - mu*(2*y*(y-1))
    return sour

if __name__ == '__main__':

    for i in range(5):
        bounds = [0, 1, 0, 1]
        Nx = 8*(2**i)
        Ny = 8*(2**i)
        lambda_ = 1
        mu = 2
        solver = Elasticity_2D_FE_solver_Dirichlet(bounds, Nx, Ny, lambda_=lambda_, mu=mu, source1_function=source1_function, source2_function=source2_function, \
                                                basis_function_type=Basis.QUADRATIC)

        # 打印误差        
        log_str = 'Nx %5d Ny %5d L_infinity: %10.6e L2: %10.6e H1: %10.6e H1-semi: %10.6e' % \
                (Nx, Ny, solver.L_infinity_error(u1_func, u2_func), solver.L2_error(u1_func, u2_func), \
                solver.H1_error(u1_func, u2_func), solver.H1_semi_error(u1_func, u2_func))
        print(log_str)

    print("\n")

        
