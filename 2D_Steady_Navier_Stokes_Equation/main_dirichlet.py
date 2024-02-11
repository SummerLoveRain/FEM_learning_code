import numpy as np
from fem import *
from solver import Steady_Navier_Stokes_2D_FE_solver_Dirichlet
from basis import Basis

def u1_func(x, y, derivative_order_x, derivative_order_y):
    '''
    解析解
    '''
    if derivative_order_x==0 and derivative_order_y==0:
        u = x**2*y**2 + np.exp(-y)
    elif derivative_order_x==1 and derivative_order_y==0:
        u = 2*x*y**2
    elif derivative_order_x==0 and derivative_order_y==1:
        u = 2*x**2*y - np.exp(-y)
    else:
        ValueError("The derivative order is not correct!")
    return u

def u2_func(x, y, derivative_order_x, derivative_order_y):
    '''
    解析解
    '''
    if derivative_order_x==0 and derivative_order_y==0:
        u = -2/3*x*y**3 + 2 - np.pi*np.sin(np.pi*x)
    elif derivative_order_x==1 and derivative_order_y==0:
        u = -2/3*y**3 - np.pi**2*np.cos(np.pi*x)
    elif derivative_order_x==0 and derivative_order_y==1:
        u = -2*x*y**2
    else:
        ValueError("The derivative order is not correct!")
    return u

def p_func(x, y, derivative_order_x, derivative_order_y):
    '''
    解析解
    '''
    if derivative_order_x==0 and derivative_order_y==0:
        u = -(2-np.pi*np.sin(np.pi*x))*np.cos(2*np.pi*y)
    elif derivative_order_x==1 and derivative_order_y==0:
        u = np.pi**2*np.cos(np.pi*x)*np.cos(2*np.pi*y)
    elif derivative_order_x==0 and derivative_order_y==1:
        u = 4*np.pi*np.sin(2*np.pi*y) - 2*np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y)
    else:
        ValueError("The derivative order is not correct!")
    return u

def source1_function(x, y, nu):
    sour = -2*nu*x**2 - 2*nu*y**2 - nu*np.exp(-y) + np.pi**2*np.cos(np.pi*x)*np.cos(2*np.pi*y)+\
            2*x*y**2*(x**2*y**2+np.exp(-y)) + (-2*x*y**3/3+2-np.pi*np.sin(np.pi*x))*(2*x**2*y-np.exp(-y))
    return sour

def source2_function(x, y, nu):
    sour = 4*nu*x*y - nu*np.pi**3*np.sin(np.pi*x) + 2*np.pi*(2-np.pi*np.sin(np.pi*x))*np.sin(2*np.pi*y)+\
            (x**2*y**2+np.exp(-y))*(-2*y**3/3-np.pi**2*np.cos(np.pi*x)) +\
            (-2*x*y**3/3+2-np.pi*np.sin(np.pi*x))*(-2*x*y**2)
    return sour

if __name__ == '__main__':

    for i in range(5):
        bounds = [0, 1, -0.25, 0]
        Nx = 8*(2**i)
        Ny = 2*(2**i)
        nu = 2
        solver = Steady_Navier_Stokes_2D_FE_solver_Dirichlet(bounds, Nx, Ny, nu=nu, source1_function=source1_function, source2_function=source2_function, \
                                                u_basis_function_type=Basis.QUADRATIC, p_basis_function_type=Basis.LINEAR)

        # 打印误差
        u1_L_infinity_error = solver.L_infinity_error(u1_func, type='u1')
        u2_L_infinity_error = solver.L_infinity_error(u2_func, type='u2')
        if u1_L_infinity_error > u2_L_infinity_error:
            L_infinity_error = u1_L_infinity_error
        else:
            L_infinity_error = u2_L_infinity_error

        u1_L2_error = solver.L2_error(u1_func, type='u1')
        u2_L2_error = solver.L2_error(u2_func, type='u2')
        u_L2_error = np.sqrt(u1_L2_error**2 + u2_L2_error**2)

        u1_H1_error = solver.H1_error(u1_func, type='u1')
        u2_H1_error = solver.H1_error(u2_func, type='u2')
        u_H1_error = np.sqrt(u1_H1_error**2 + u2_H1_error**2)

        u1_H1_semi_error = solver.H1_semi_error(u1_func, type='u1')
        u2_H1_semi_error = solver.H1_semi_error(u2_func, type='u2')
        u_H1_semi_error = np.sqrt(u1_H1_semi_error**2 + u2_H1_semi_error**2)
        log_str = 'Nx %5d Ny %5d L_infinity: %10.6e L2: %10.6e H1: %10.6e H1-semi: %10.6e' % \
                (Nx, Ny, L_infinity_error, u_L2_error, u_H1_error, u_H1_semi_error)
        print(log_str)
        log_str = 'Nx %5d Ny %5d L_infinity: %10.6e L2: %10.6e H1: %10.6e H1-semi: %10.6e' % \
                (Nx, Ny, solver.L_infinity_error(p_func, type='p'), solver.L2_error(p_func, type='p'), \
                solver.H1_error(p_func, type='p'), solver.H1_semi_error(p_func, type='p'))
        print(log_str)
        
        print("\n")

    print("\n")

        
