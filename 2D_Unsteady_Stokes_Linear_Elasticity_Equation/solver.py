import math
import numpy as np
import scipy
import scipy.sparse as sp
from basis import local_linear_triangular_2D_basis, local_quadratic_triangular_2D_basis, Basis
from fem import *
from integration import integration, Gaussian_Integral_triangular_2D

class Base_2D_Solver:
    def __init__(self, bounds, Nx, Ny, Nt, theta, nu, u_basis_function_type, p_basis_function_type, Gaussian_Integral_triangular_2D_N=10):
        # the bounds of problems in a rectangular domain
        self.bounds = bounds        
        self.a = self.bounds[0]
        self.b = self.bounds[1]
        self.c = self.bounds[2]
        self.d = self.bounds[3]
        self.initial_time = self.bounds[4]
        self.end_time = self.bounds[5]
        self.nu = nu
        # the number of partition on each axis
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.dt = (self.end_time-self.initial_time)/self.Nt
        # theta=0: forward Euler.
        # theta=1: backward Euler.
        # theta=0.5: Crank-Nicolson.
        self.theta = theta
        # the number of elements
        self.Nb = 2*Nx*Ny
        # the number of mesh nodes
        self.Nm = (Nx+1)*(Ny+1)
        # the type of trial and test basis function
        self.u_basis_function_type = u_basis_function_type
        self.p_basis_function_type = p_basis_function_type
        # the order used in Gaussian Integrals
        self.Gaussian_Integral_triangular_2D_N = Gaussian_Integral_triangular_2D_N

        if self.u_basis_function_type == Basis.LINEAR:
            self.u_basis_function = local_linear_triangular_2D_basis
        elif self.u_basis_function_type == Basis.QUADRATIC:
            self.u_basis_function = local_quadratic_triangular_2D_basis
        else:
            ValueError("The basis_function_type is not correct!")

        if self.p_basis_function_type == Basis.LINEAR:
            self.p_basis_function = local_linear_triangular_2D_basis
        elif self.p_basis_function_type == Basis.QUADRATIC:
            self.p_basis_function = local_quadratic_triangular_2D_basis
        else:
            ValueError("The basis_function_type is not correct!")
    
    def generate_P_T(self):
        # P is an information matrix consisting of the coordinates of all mesh nodes
        # T is an information matrix consisting of the global node indices of the mesh nodes of all the mesh elements
        P, T = self.generate_Pb_Tb_linear()
        return P, T
    
    def repeat_tile(self, x, y):
        return np.transpose([np.repeat(x, len(y)),
                            np.tile(y, len(x))])

    def generate_Pb_Tb_linear(self):
        # [P, T] is the same as [Pb, Tb] in the linear case
        coord_x = np.linspace(self.a, self.b, self.Nx+1)
        coord_y = np.linspace(self.c, self.d, self.Ny+1)
        Pb = self.repeat_tile(coord_x, coord_y)
        Tb = np.zeros((self.Nb, 3), dtype=int)
        for i in range(self.Nb):
            # the index of elements
            n = i + 1
            # the column index
            times = n//(2*self.Ny)
            remainder = n%(2*self.Ny)
            if remainder > 0:
                c = times + 1
            else:
                c = times
            # the row index
            r = (n - 2*self.Ny*(c-1) + 1)//2
            first_number = (c-1)*(self.Ny+1) + r - 1
            # odd number, even number
            if n%2 == 1:
                second_number = first_number + self.Ny + 1
                Tb[i, 0] = first_number
                Tb[i, 1] = second_number
                Tb[i, 2] = first_number + 1
            else:
                first_number += 1
                second_number = first_number + self.Ny
                Tb[i, 0] = first_number
                Tb[i, 1] = second_number
                Tb[i, 2] = second_number + 1
        return Pb, Tb
    
    def generate_Pb_Tb_quadratic(self):
        # Pb is an information matrix consisting of the coordinates of all finite element nodes.
        # Tb is an information matrix consisting of the global node indices of the finite element nodes of all the mesh elements.
        coord_x = np.linspace(self.a, self.b, 2*self.Nx+1)
        coord_y = np.linspace(self.c, self.d, 2*self.Ny+1)
        Pb = self.repeat_tile(coord_x, coord_y)
        Tb = np.zeros((self.Nb, 6), dtype=int)
        for i in range(self.Nb):
            # the index of elements
            n = i + 1
            # the column index
            times = n//(2*self.Ny)
            remainder = n%(2*self.Ny)
            if remainder > 0:
                c = times + 1
            else:
                c = times
            # the row index
            r = (n - 2*self.Ny*(c-1) + 1)//2
            first_number = (c-1)*(2*(2*self.Ny+1)) + (2*r-1) - 1
            # odd number, even number
            if n%2 == 1:
                second_number = first_number + 2*(2*self.Ny+1)
                fourth_number = first_number + (2*self.Ny + 1)
                Tb[i, 0] = first_number
                Tb[i, 1] = second_number
                Tb[i, 2] = first_number + 2
                Tb[i, 3] = fourth_number
                Tb[i, 4] = fourth_number + 1
                Tb[i, 5] = fourth_number - 2*self.Ny
            else:
                first_number += 2
                fourth_number = first_number + 2*self.Ny
                fifth_number = fourth_number + 2*self.Ny + 1
                Tb[i, 0] = first_number
                Tb[i, 1] = fifth_number - 1
                Tb[i, 2] = fifth_number + 1
                Tb[i, 3] = fourth_number
                Tb[i, 4] = fifth_number
                Tb[i, 5] = fourth_number + 1
        return Pb, Tb
    
    def generate_Pb_Tb(self, basis_function_type=Basis.LINEAR):
        if basis_function_type == Basis.LINEAR:
            Pb, Tb = self.generate_Pb_Tb_linear()
        elif basis_function_type == Basis.QUADRATIC:
            Pb, Tb = self.generate_Pb_Tb_quadratic()
        return Pb, Tb
    
    def is_in_triangle(self, x, y, vertices):
        x1 = vertices[0, 0]
        y1 = vertices[0, 1]
        x2 = vertices[1, 0]
        y2 = vertices[1, 1]
        x3 = vertices[2, 0]
        y3 = vertices[2, 1]
        
        def getSideLength(x1, y1, x2, y2):
            a = np.abs(x2 - x1)
            b = np.abs(y2 - y1)
            return np.sqrt(a*a + b*b)
    
        def getArea(x1, y1, x2, y2, x3, y3):
            a = getSideLength(x1, y1, x2, y2)
            b = getSideLength(x1, y1, x3, y3)
            c = getSideLength(x2, y2, x3, y3)
            p = (a + b + c) / 2
            area_square = p * (p-a) * (p-b) * (p-c)
            if area_square < 1e-15:
                return 0
            else:
                return np.sqrt(area_square)
    
        area1 = getArea(x1, y1, x2, y2, x, y)
        # print(area1)
        area2 = getArea(x1, y1, x3, y3, x, y)
        # print(area2)
        area3 = getArea(x2, y2, x3, y3, x, y)
        # print(area3)
        allArea = getArea(x1, y1, x2, y2, x3, y3)
        # print(allArea)
        return (area1 + area2 + area3) <= allArea
    
    def solve(self):
        '''
        return uh
        '''
        print("solve for uh!")

    def L_infinity_error(self, u_func, type='u1'):
        '''
        u_func: The function of exact solution
        compute the L-infinity error
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Nb = T.shape[0]
        if type == 'u1':
            basis_function = self.u_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.u_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[:Ng]
        elif type == 'u2':
            basis_function = self.u_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.u_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[Ng:2*Ng]
        else:
            basis_function = self.p_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.p_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[-Ng:]

        # x = Pb[:, 0:1]
        # y = Pb[:, 1:2]
        # u = u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=0)

        error=0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def uh_func(x, y):
                u = 0.0
                for j in range(Nl):
                    u += uh[Tb[i, j]]*basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return u
            f_integral = lambda x ,y: np.abs(u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=0) - uh_func(x, y))
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N, integration_type=integration.MAX)
            if integral>error:
                error = integral
        return error
        

    def L2_error(self, u_func, type='u1'):
        '''
        u_func: The function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Nb = T.shape[0]
        if type == 'u1':
            basis_function = self.u_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.u_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[:Ng]
        elif type == 'u2':
            basis_function = self.u_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.u_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[Ng:2*Ng]
        else:
            basis_function = self.p_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.p_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[-Ng:]
        error = 0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def uh_func(x, y):
                u = 0.0
                for j in range(Nl):
                    u += uh[Tb[i, j]]*basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return u
            f_integral = lambda x ,y: (u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=0) - uh_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error
    
    def H1_error(self, u_func, type='u1'):
        '''
        u_func: The function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Nb = T.shape[0]
        if type == 'u1':
            basis_function = self.u_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.u_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[:Ng]
        elif type == 'u2':
            basis_function = self.u_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.u_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[Ng:2*Ng]
        else:
            basis_function = self.p_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.p_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[-Ng:]
        error = 0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def uh_func(x, y):
                u = 0.0
                for j in range(Nl):
                    u += uh[Tb[i, j]]*basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return u
            def uh_x_func(x, y):
                u = 0.0
                for j in range(Nl):
                    ux = basis_function(x, y, vertices, j, derivative_order_x=1, derivative_order_y=0)
                    u += uh[Tb[i, j]] * ux
                return u
            def uh_y_func(x, y):
                u = 0.0
                for j in range(Nl):
                    uy = basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=1)
                    u += uh[Tb[i, j]] * uy
                return u
            f_integral = lambda x, y : (u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=0) - uh_func(x, y))**2 +\
                  (u_func(x, y, self.end_time, derivative_order_x=1, derivative_order_y=0) - uh_x_func(x, y))**2 +\
                  (u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=1) - uh_y_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error
    
    
    def H1_semi_error(self, u_func, type='u1'):
        '''
        u_func: The function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Nb = T.shape[0]
        if type == 'u1':
            basis_function = self.u_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.u_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[:Ng]
        elif type == 'u2':
            basis_function = self.u_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.u_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[Ng:2*Ng]
        else:
            basis_function = self.p_basis_function
            Pb, Tb = self.generate_Pb_Tb(basis_function_type=self.p_basis_function_type)
            Nl = Tb.shape[1]
            # the number of nodes
            Ng = np.max(Tb) + 1
            uh = self.uh[-Ng:]
        error = 0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def uh_x_func(x, y):
                u = 0.0
                for j in range(Nl):
                    ux = basis_function(x, y, vertices, j, derivative_order_x=1, derivative_order_y=0)
                    u += uh[Tb[i, j]] * ux
                return u
            def uh_y_func(x, y):
                u = 0.0
                for j in range(Nl):
                    uy = basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=1)
                    u += uh[Tb[i, j]] * uy
                return u
            f_integral = lambda x, y : (u_func(x, y, self.end_time, derivative_order_x=1, derivative_order_y=0) - uh_x_func(x, y))**2 +\
                  (u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=1) - uh_y_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error


class Unsteady_Stokes_Linear_Elasticity_2D_FE_solver_Dirichlet(Base_2D_Solver):
    def __init__(self, bounds, Nx, Ny, Nt, theta, nu, u_basis_function_type, p_basis_function_type, source1_function, source2_function, Gaussian_Integral_triangular_2D_N=10):
        '''
        bounds: the bounds of problems in a rectangular domain
        Nx, Ny: the number of partition on each axis
        the order used in Gaussian Integrals
        basis_function_type: linear, quadratic
        '''
        super(Unsteady_Stokes_Linear_Elasticity_2D_FE_solver_Dirichlet, self).__init__(bounds, Nx, Ny, Nt, theta, nu, u_basis_function_type, p_basis_function_type, Gaussian_Integral_triangular_2D_N)
        self.source1_function = source1_function
        self.source2_function = source2_function
        
        self.uh = None
    
    def initial_condition1(self, x, y):
        # the value on the boundary at initial time
        u = x**2*y**2 + np.exp(-y)
        return u
    
    def initial_condition2(self, x, y):
        # the value on the boundary at initial time
        u = -2/3*x*y**3 + 2 - np.pi*np.sin(np.pi*x)
        return u
    
    def initial_condition3(self, x, y):
        # the value on the boundary at initial time
        u = -(2-np.pi*np.sin(np.pi*x))*np.cos(2*np.pi*y)
        return u
    
    def dirichlet_boundary1(self, x, y, t):
        if np.abs(x-self.a)<1e-15:
            u = np.exp(-y)*np.cos(2*np.pi*t)
        elif np.abs(x-self.b)<1e-15:
            u = (y**2 + np.exp(-y))*np.cos(2*np.pi*t)
        elif np.abs(y-self.c)<1e-15:
            u = (1/16*x**2 + np.exp(0.25))*np.cos(2*np.pi*t)
        elif np.abs(y-self.d)<1e-15:
            u = 1*np.cos(2*np.pi*t)
        else:
            ValueError('No boundary edges!')
        return u
    
    def dirichlet_boundary2(self, x, y, t):
        if np.abs(x-self.a)<1e-15:
            u = 2*np.cos(2*np.pi*t)
        elif np.abs(x-self.b)<1e-15:
            u = (-2/3*y**3 + 2)*np.cos(2*np.pi*t)
        elif np.abs(y-self.c)<1e-15:
            u = (1/96*x + 2 - np.pi*np.sin(np.pi*x))*np.cos(2*np.pi*t)
        elif np.abs(y-self.d)<1e-15:
            u = (2 - np.pi*np.sin(np.pi*x))*np.cos(2*np.pi*t)
        else:
            ValueError('No boundary edges!')
        return u
    
    def dirichlet_boundary3(self, x, y, t):
        if np.abs(x-self.a)<1e-15:
            u = -(2-np.pi*np.sin(np.pi*self.a))*np.cos(2*np.pi*y)*np.cos(2*np.pi*t)
        elif np.abs(x-self.b)<1e-15:
            u = -(2-np.pi*np.sin(np.pi*self.b))*np.cos(2*np.pi*y)*np.cos(2*np.pi*t)
        elif np.abs(y-self.c)<1e-15:
            u = -(2-np.pi*np.sin(np.pi*x))*np.cos(2*np.pi*self.c)*np.cos(2*np.pi*t)
        elif np.abs(y-self.d)<1e-15:
            u = -(2-np.pi*np.sin(np.pi*x))*np.cos(2*np.pi*self.d)*np.cos(2*np.pi*t)
        else:
            ValueError('No boundary edges!')
        return u
    
    def generate_boundary_edges(self, basis_function_type):
        #boundary_edges[i,0] is the type of the k-th boundary edge ek:
        # Dirichlet, Neumann, Robin
        #boundary_edges[i,1] is the index of the element which contains the k-th boundary edge ek
        # Each boundary edge has two end nodes. We index them as the first and the second counterclock wise along the boundary.
        # boundary_edges[i, 2] is the global node index of the first end node of the k-th boundary edge ek.
        # boundary_edges[i, 3] is the global node index of the second end node of the k-th boundary edge ek.
        if basis_function_type == Basis.LINEAR:
            boundary_edges = self.generate_boundary_linear()
        elif basis_function_type == Basis.QUADRATIC:
            boundary_edges = self.generate_boundary_quadratic()
        return boundary_edges

    def generate_boundary_linear(self):
        boundary_edges = np.zeros((2*(self.Nx+self.Ny), 4),dtype=int)
        _, Tb = self.generate_Pb_Tb()
        # boundary edge on the x-axis
        for i in range(2*(self.Nx+self.Ny)):
            if i<self.Nx:
                # number of the element
                n = i*(2*self.Ny) + 1
                n = n-1
                boundary_edges[i, 0] = Boundary.DIRICHLET_BOUNDARY
                boundary_edges[i, 1] = n
                boundary_edges[i, 2] = Tb[n, 0]
                boundary_edges[i, 3] = Tb[n, 1]
            elif i>=self.Nx and i<self.Nx+self.Ny:
                # number of the element
                n = (self.Nx-1)*(2*self.Ny) + 2*(i-self.Nx + 1)
                n = n-1
                boundary_edges[i, 0] = Boundary.DIRICHLET_BOUNDARY
                boundary_edges[i, 1] = n
                boundary_edges[i, 2] = Tb[n, 1]
                boundary_edges[i, 3] = Tb[n, 2]
            elif i>=self.Nx+self.Ny and i<2*self.Nx+self.Ny:
                # number of the element
                n = (self.Nx-(i-self.Nx-self.Ny))*2*self.Ny
                n = n-1
                boundary_edges[i, 0] = Boundary.DIRICHLET_BOUNDARY
                boundary_edges[i, 1] = n
                boundary_edges[i, 2] = Tb[n, 2]
                boundary_edges[i, 3] = Tb[n, 0]
            elif i>=2*self.Nx+self.Ny and i<2*self.Nx+2*self.Ny:
                # number of the element
                n = 2*(self.Ny-(i-self.Nx-self.Nx-self.Ny))-1
                n = n-1
                boundary_edges[i, 0] = Boundary.DIRICHLET_BOUNDARY
                boundary_edges[i, 1] = n
                boundary_edges[i, 2] = Tb[n, 2]
                boundary_edges[i, 3] = Tb[n, 0]
        return boundary_edges
    
    
    def generate_boundary_quadratic(self):
        boundary_edges = np.zeros((2*(self.Nx+self.Ny), 5),dtype=int)
        _, Tb = self.generate_Pb_Tb(basis_function_type=Basis.QUADRATIC)
        # boundary edge on the x-axis
        for i in range(2*(self.Nx+self.Ny)):
            if i<self.Nx:
                # number of the element
                n = i*(2*self.Ny) + 1
                n = n-1
                boundary_edges[i, 0] = Boundary.DIRICHLET_BOUNDARY
                boundary_edges[i, 1] = n
                boundary_edges[i, 2] = Tb[n, 0]
                boundary_edges[i, 3] = Tb[n, 1]
                boundary_edges[i, 4] = Tb[n, 3]
            elif i>=self.Nx and i<self.Nx+self.Ny:
                # number of the element
                n = (self.Nx-1)*(2*self.Ny) + 2*(i-self.Nx + 1)
                n = n-1
                boundary_edges[i, 0] = Boundary.DIRICHLET_BOUNDARY
                boundary_edges[i, 1] = n
                boundary_edges[i, 2] = Tb[n, 1]
                boundary_edges[i, 3] = Tb[n, 2]
                boundary_edges[i, 4] = Tb[n, 4]
            elif i>=self.Nx+self.Ny and i<2*self.Nx+self.Ny:
                # number of the element
                n = (self.Nx-(i-self.Nx-self.Ny))*2*self.Ny
                n = n-1
                boundary_edges[i, 0] = Boundary.DIRICHLET_BOUNDARY
                boundary_edges[i, 1] = n
                boundary_edges[i, 2] = Tb[n, 2]
                boundary_edges[i, 3] = Tb[n, 0]
                boundary_edges[i, 4] = Tb[n, 5]
            elif i>=2*self.Nx+self.Ny and i<2*self.Nx+2*self.Ny:
                # number of the element
                n = 2*(self.Ny-(i-self.Nx-self.Nx-self.Ny))-1
                n = n-1
                boundary_edges[i, 0] = Boundary.DIRICHLET_BOUNDARY
                boundary_edges[i, 1] = n
                boundary_edges[i, 2] = Tb[n, 2]
                boundary_edges[i, 3] = Tb[n, 0]
                boundary_edges[i, 4] = Tb[n, 5]
        return boundary_edges
    
    def solve(self):
        P, T = self.generate_P_T()
        u_Pb, u_Tb = self.generate_Pb_Tb(self.u_basis_function_type)
        p_Pb, p_Tb = self.generate_Pb_Tb(self.p_basis_function_type)
        boundary_edges = self.generate_boundary_edges(self.u_basis_function_type)
        boundary_edges2 = self.generate_boundary_edges(self.p_basis_function_type)
        
        # the number of nodes
        u_Ng = np.max(u_Tb) + 1
        p_Ng = np.max(p_Tb) + 1
        # the number of local basis functions
        u_Nl = u_Tb.shape[1]
        # the number of finite elements
        u_Nb = u_Tb.shape[0]

        A1 = integrate_trial_test_basis_function_2D(P, T, u_Pb, u_Tb, u_Pb, u_Tb, coefficient_function=lambda x, y: self.nu, \
                        trial_basis_function=self.u_basis_function, test_basis_function=self.u_basis_function, \
                        trial_derivative_order=[1, 0], test_derivative_order=[1, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A2 = integrate_trial_test_basis_function_2D(P, T, u_Pb, u_Tb, u_Pb, u_Tb, coefficient_function=lambda x, y: self.nu, \
                        trial_basis_function=self.u_basis_function, test_basis_function=self.u_basis_function, \
                        trial_derivative_order=[0, 1], test_derivative_order=[0, 1], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A3 = integrate_trial_test_basis_function_2D(P, T, u_Pb, u_Tb, u_Pb, u_Tb, coefficient_function=lambda x, y: self.nu, \
                        trial_basis_function=self.u_basis_function, test_basis_function=self.u_basis_function, \
                        trial_derivative_order=[1, 0], test_derivative_order=[0, 1], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A4 = integrate_trial_test_basis_function_2D(P, T, u_Pb, u_Tb, u_Pb, u_Tb, coefficient_function=lambda x, y: self.nu, \
                        trial_basis_function=self.u_basis_function, test_basis_function=self.u_basis_function, \
                        trial_derivative_order=[0, 1], test_derivative_order=[1, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A5 = integrate_trial_test_basis_function_2D(P, T, p_Pb, p_Tb, u_Pb, u_Tb, coefficient_function=lambda x, y: -1, \
                        trial_basis_function=self.p_basis_function, test_basis_function=self.u_basis_function, \
                        trial_derivative_order=[0, 0], test_derivative_order=[1, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A6 = integrate_trial_test_basis_function_2D(P, T, p_Pb, p_Tb, u_Pb, u_Tb, coefficient_function=lambda x, y: -1, \
                        trial_basis_function=self.p_basis_function, test_basis_function=self.u_basis_function, \
                        trial_derivative_order=[0, 0], test_derivative_order=[0, 1], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A7 = integrate_trial_test_basis_function_2D(P, T, u_Pb, u_Tb, p_Pb, p_Tb, coefficient_function=lambda x, y: -1, \
                        trial_basis_function=self.u_basis_function, test_basis_function=self.p_basis_function, \
                        trial_derivative_order=[1, 0], test_derivative_order=[0, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A8 = integrate_trial_test_basis_function_2D(P, T, u_Pb, u_Tb, p_Pb, p_Tb, coefficient_function=lambda x, y: -1, \
                        trial_basis_function=self.u_basis_function, test_basis_function=self.p_basis_function, \
                        trial_derivative_order=[0, 1], test_derivative_order=[0, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        # A9 = integrate_trial_test_basis_function_2D(P, T, p_Pb, p_Tb, p_Pb, p_Tb, coefficient_function=lambda x, y: 1, \
        #                 trial_basis_function=self.p_basis_function, test_basis_function=self.p_basis_function, \
        #                 trial_derivative_order=[0, 0], test_derivative_order=[0, 0], \
        #                 Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A11 = 2*A1 + A2
        A12 = A3
        A13 = A5
        A21 = A4
        A22 = 2*A2 + A1
        A23 = A6
        A31 = A7
        A32 = A8
        # A33 = A9
        A33 = None
        A = sp.bmat([[A11, A12, A13],
                     [A21, A22, A23],
                     [A31, A32, A33]], format='lil')
        # A, b = treat_Dirichlet_boundary_2D(u_Pb, u_Tb, A, b, boundary_edges, self.dirichlet_boundary1, self.dirichlet_boundary2)
        # A, b = treat_Dirichlet_boundary_2D2(P, T, u_Pb, u_Tb, A, b, boundary_edges, self.dirichlet_boundary1, self.dirichlet_boundary2, boundary_edges2, self.dirichlet_boundary3)
        
        Me = assemble_matrix_M_2D(P, T, u_Pb, u_Tb, self.u_basis_function, self.Gaussian_Integral_triangular_2D_N)
        M3 = lil_matrix((p_Ng, p_Ng))
        M = sp.bmat([[Me, None, None],
                     [None, Me, None],
                     [None, None, M3]], format='lil')
        A_tilde = M/self.dt + self.theta*A
        A_tilde = treat_Dirichlet_boundary_2D_A(P, T, u_Pb, u_Tb, A_tilde, boundary_edges, boundary_edges2)
        x = u_Pb[:, 0:1]
        y = u_Pb[:, 1:2]
        u1_t = self.initial_condition1(x, y)
        u2_t = self.initial_condition2(x, y)

        x = p_Pb[:, 0:1]
        y = p_Pb[:, 1:2]
        p_t = self.initial_condition3(x, y)
        u_t = np.concatenate((u1_t, u2_t, p_t), axis=0)

 
        # define a sparse matrix
        b3 = lil_matrix((p_Ng, 1))

        for i in range(self.Nt):
            current_time = self.initial_time + (i+1)*self.dt


            b_t1 = assemble_vector_b_2D(P, T, u_Tb, current_time-self.dt, self.nu, self.source1_function, self.u_basis_function, self.Gaussian_Integral_triangular_2D_N)
            b_t2 = assemble_vector_b_2D(P, T, u_Tb, current_time-self.dt, self.nu, self.source2_function, self.u_basis_function, self.Gaussian_Integral_triangular_2D_N)       
            b_t = sp.bmat([[b_t1],
                        [b_t2],
                        [b3]], format='lil')            

            b_tp1 = assemble_vector_b_2D(P, T, u_Tb, current_time, self.nu, self.source1_function, self.u_basis_function, self.Gaussian_Integral_triangular_2D_N)
            b_tp2 = assemble_vector_b_2D(P, T, u_Tb, current_time, self.nu, self.source2_function, self.u_basis_function, self.Gaussian_Integral_triangular_2D_N)
            b_tp = sp.bmat([[b_tp1],
                        [b_tp2],
                        [b3]], format='lil')
            b_tilde = self.theta*b_tp + (1-self.theta)*b_t + (M/self.dt - (1-self.theta)*A).dot(u_t)
            b_tilde = treat_Dirichlet_boundary_2D_b(P, T, u_Pb, u_Tb, b_tilde, current_time, boundary_edges, self.dirichlet_boundary1, self.dirichlet_boundary2, boundary_edges2, self.dirichlet_boundary3)
            u_tp = scipy.sparse.linalg.spsolve(A_tilde.tocsc(), b_tilde)
            u_tp = np.reshape(u_tp, u_t.shape)
            u_t = u_tp

        uh = u_tp
        
        return uh

