import math
import numpy as np
import scipy
import scipy.sparse as sp
from basis import local_linear_triangular_2D_basis, local_quadratic_triangular_2D_basis, Basis
from fem import *
from integration import integration, Gaussian_Integral_triangular_2D

class Base_2D_Solver:
    def __init__(self, bounds, Nx, Ny, lambda_, mu, basis_function_type, Gaussian_Integral_triangular_2D_N=10):
        # the bounds of problems in a rectangular domain
        self.bounds = bounds        
        self.a = self.bounds[0]
        self.b = self.bounds[1]
        self.c = self.bounds[2]
        self.d = self.bounds[3]
        self.lambda_ = lambda_
        self.mu = mu
        # the number of partition on each axis
        self.Nx = Nx
        self.Ny = Ny
        # the number of elements
        self.Nb = 2*Nx*Ny
        # the number of mesh nodes
        self.Nm = (Nx+1)*(Ny+1)
        # the type of trial and test basis function
        self.basis_function_type = basis_function_type
        # the order used in Gaussian Integrals
        self.Gaussian_Integral_triangular_2D_N = Gaussian_Integral_triangular_2D_N

        if self.basis_function_type == Basis.LINEAR:
            self.basis_function = local_linear_triangular_2D_basis
        elif self.basis_function_type == Basis.QUADRATIC:
            self.basis_function = local_quadratic_triangular_2D_basis
        else:
            ValueError("The basis_function_type_trial is not correct!")
    
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
    
    def generate_Pb_Tb(self):
        if self.basis_function_type == Basis.LINEAR:
            Pb, Tb = self.generate_Pb_Tb_linear()
        elif self.basis_function_type == Basis.QUADRATIC:
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

    def L_infinity_error(self, u1_func, u2_func):
        '''
        u_func: The function of exact solution
        compute the L-infinity error
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb, Tb = self.generate_Pb_Tb()
        # the number of nodes
        Ng = np.max(Tb) + 1
        u1h = self.uh[:Ng]
        u2h = self.uh[Ng:]
        Nl = Tb.shape[1]
        Nb = T.shape[0]
        error=0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def u1h_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uh += u1h[Tb[i, j]]*self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return uh
            f_integral = lambda x ,y: np.abs(u1_func(x, y, derivative_order_x=0, derivative_order_y=0) - u1h_func(x, y))
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N, integration_type=integration.MAX)
            if integral>error:
                error = integral   
            def u2h_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uh += u2h[Tb[i, j]]*self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return uh             
            f_integral = lambda x ,y: np.abs(u2_func(x, y, derivative_order_x=0, derivative_order_y=0) - u2h_func(x, y))
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N, integration_type=integration.MAX)
            if integral>error:
                error = integral   
        return error
        

    def L2_error(self, u1_func, u2_func):
        '''
        u_func: The function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb, Tb = self.generate_Pb_Tb()
        # N_trial:  the total number of the trial functions in the local space
        Nl = Tb.shape[1]

        Nb = T.shape[0]
        # the number of nodes
        Ng = np.max(Tb) + 1
        u1h = self.uh[:Ng]
        u2h = self.uh[Ng:]
        error = 0.0

        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def u1h_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uh += u1h[Tb[i, j]]*self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return uh
            f_integral = lambda x ,y: (u1_func(x, y, derivative_order_x=0, derivative_order_y=0) - u1h_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral

            def u2h_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uh += u2h[Tb[i, j]]*self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return uh
            f_integral = lambda x ,y: (u2_func(x, y, derivative_order_x=0, derivative_order_y=0) - u2h_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error
    
    def H1_error(self, u1_func, u2_func):
        '''
        u_func: The function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb, Tb = self.generate_Pb_Tb()
        # N_trial:  the total number of the trial functions in the local space
        Nl = Tb.shape[1]

        Nb = T.shape[0]
        # the number of nodes
        Ng = np.max(Tb) + 1
        u1h = self.uh[:Ng]
        u2h = self.uh[Ng:]
        error = 0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def u1h_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uh += u1h[Tb[i, j]]*self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return uh
            def u1h_x_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    ux = self.basis_function(x, y, vertices, j, derivative_order_x=1, derivative_order_y=0)
                    uh += u1h[Tb[i, j]] * ux
                return uh
            def u1h_y_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uy = self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=1)
                    uh += u1h[Tb[i, j]] * uy
                return uh
            f_integral = lambda x, y : (u1_func(x, y, derivative_order_x=0, derivative_order_y=0) - u1h_func(x, y))**2 +\
                  (u1_func(x, y, derivative_order_x=1, derivative_order_y=0) - u1h_x_func(x, y))**2 +\
                  (u1_func(x, y, derivative_order_x=0, derivative_order_y=1) - u1h_y_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral

            
            def u2h_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uh += u2h[Tb[i, j]]*self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return uh
            def u2h_x_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    ux = self.basis_function(x, y, vertices, j, derivative_order_x=1, derivative_order_y=0)
                    uh += u2h[Tb[i, j]] * ux
                return uh
            def u2h_y_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uy = self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=1)
                    uh += u2h[Tb[i, j]] * uy
                return uh
            f_integral = lambda x, y : (u2_func(x, y, derivative_order_x=0, derivative_order_y=0) - u2h_func(x, y))**2 +\
                  (u2_func(x, y, derivative_order_x=1, derivative_order_y=0) - u2h_x_func(x, y))**2 +\
                  (u2_func(x, y, derivative_order_x=0, derivative_order_y=1) - u2h_y_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error
    
    
    def H1_semi_error(self, u1_func, u2_func):
        '''
        u_func: The function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb, Tb = self.generate_Pb_Tb()
        # N_trial:  the total number of the trial functions in the local space
        Nl = Tb.shape[1]

        Nb = T.shape[0]
        # the number of nodes
        Ng = np.max(Tb) + 1
        u1h = self.uh[:Ng]
        u2h = self.uh[Ng:]
        error = 0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def u1h_x_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    ux = self.basis_function(x, y, vertices, j, derivative_order_x=1, derivative_order_y=0)
                    uh += u1h[Tb[i, j]] * ux
                return uh
            def u1h_y_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uy = self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=1)
                    uh += u1h[Tb[i, j]] * uy
                return uh
            f_integral = lambda x, y : (u1_func(x, y, derivative_order_x=1, derivative_order_y=0) - u1h_x_func(x, y))**2 +\
                  (u1_func(x, y, derivative_order_x=0, derivative_order_y=1) - u1h_y_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral

            
            def u2h_x_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    ux = self.basis_function(x, y, vertices, j, derivative_order_x=1, derivative_order_y=0)
                    uh += u2h[Tb[i, j]] * ux
                return uh
            def u2h_y_func(x, y):
                uh = 0.0
                for j in range(Nl):
                    uy = self.basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=1)
                    uh += u2h[Tb[i, j]] * uy
                return uh
            f_integral = lambda x, y : (u2_func(x, y, derivative_order_x=1, derivative_order_y=0) - u2h_x_func(x, y))**2 +\
                  (u2_func(x, y, derivative_order_x=0, derivative_order_y=1) - u2h_y_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error



class Elasticity_2D_FE_solver_Dirichlet(Base_2D_Solver):
    def __init__(self, bounds, Nx, Ny, lambda_, mu, basis_function_type, source1_function, source2_function, Gaussian_Integral_triangular_2D_N=10):
        '''
        bounds: the bounds of problems in a rectangular domain
        Nx, Ny: the number of partition on each axis
        the order used in Gaussian Integrals
        basis_function_type: linear, quadratic
        '''
        super(Elasticity_2D_FE_solver_Dirichlet, self).__init__(bounds, Nx, Ny, lambda_, mu, basis_function_type, Gaussian_Integral_triangular_2D_N)
        self.source1_function = source1_function
        self.source2_function = source2_function
        
        self.uh = None
    
    def dirichlet_boundary1(self, x, y):
        if np.abs(x-self.a)<1e-15:
            u = np.sin(np.pi*self.a)*np.sin(np.pi*y)
        elif np.abs(x-self.b)<1e-15:
            u = np.sin(np.pi*self.b)*np.sin(np.pi*y)
        elif np.abs(y-self.c)<1e-15:
            u = np.sin(np.pi*x)*np.sin(np.pi*self.c)
        elif np.abs(y-self.d)<1e-15:
            u = np.sin(np.pi*x)*np.sin(np.pi*self.d)
        else:
            ValueError('No boundary edges!')
        return u
    
    def dirichlet_boundary2(self, x, y):
        if np.abs(x-self.a)<1e-15:
            u = self.a*(self.a-1)*y*(y-1)
        elif np.abs(x-self.b)<1e-15:
            u = self.b*(self.b-1)*y*(y-1)
        elif np.abs(y-self.c)<1e-15:
            u = x*(x-1)*self.c*(self.c-1)
        elif np.abs(y-self.d)<1e-15:
            u = x*(x-1)*self.d*(self.d-1)
        else:
            ValueError('No boundary edges!')
        return u
    
    def generate_boundary_edges(self):
        #boundary_edges[i,0] is the type of the k-th boundary edge ek:
        # Dirichlet, Neumann, Robin
        #boundary_edges[i,1] is the index of the element which contains the k-th boundary edge ek
        # Each boundary edge has two end nodes. We index them as the first and the second counterclock wise along the boundary.
        # boundary_edges[i, 2] is the global node index of the first end node of the k-th boundary edge ek.
        # boundary_edges[i, 3] is the global node index of the second end node of the k-th boundary edge ek.
        if self.basis_function_type == Basis.LINEAR:
            boundary_edges = self.generate_boundary_linear()
        elif self.basis_function_type == Basis.QUADRATIC:
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
        Pb, Tb = self.generate_Pb_Tb()
        boundary_edges = self.generate_boundary_edges()
        
        A1 = integrate_trial_test_basis_function_2D(P, T, Pb, Tb, coefficient_function=lambda x, y: self.lambda_, \
                        trial_basis_function=self.basis_function, test_basis_function=self.basis_function, \
                        trial_derivative_order=[1, 0], test_derivative_order=[1, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A2 = integrate_trial_test_basis_function_2D(P, T, Pb, Tb, coefficient_function=lambda x, y: self.mu, \
                        trial_basis_function=self.basis_function, test_basis_function=self.basis_function, \
                        trial_derivative_order=[1, 0], test_derivative_order=[1, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A3 = integrate_trial_test_basis_function_2D(P, T, Pb, Tb, coefficient_function=lambda x, y: self.mu, \
                        trial_basis_function=self.basis_function, test_basis_function=self.basis_function, \
                        trial_derivative_order=[0, 1], test_derivative_order=[0, 1], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A4 = integrate_trial_test_basis_function_2D(P, T, Pb, Tb, coefficient_function=lambda x, y: self.lambda_, \
                        trial_basis_function=self.basis_function, test_basis_function=self.basis_function, \
                        trial_derivative_order=[0, 1], test_derivative_order=[1, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A5 = integrate_trial_test_basis_function_2D(P, T, Pb, Tb, coefficient_function=lambda x, y: self.mu, \
                        trial_basis_function=self.basis_function, test_basis_function=self.basis_function, \
                        trial_derivative_order=[1, 0], test_derivative_order=[0, 1], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A6 = integrate_trial_test_basis_function_2D(P, T, Pb, Tb, coefficient_function=lambda x, y: self.lambda_, \
                        trial_basis_function=self.basis_function, test_basis_function=self.basis_function, \
                        trial_derivative_order=[1, 0], test_derivative_order=[0, 1], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A7 = integrate_trial_test_basis_function_2D(P, T, Pb, Tb, coefficient_function=lambda x, y: self.mu, \
                        trial_basis_function=self.basis_function, test_basis_function=self.basis_function, \
                        trial_derivative_order=[0, 1], test_derivative_order=[1, 0], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A8 = integrate_trial_test_basis_function_2D(P, T, Pb, Tb, coefficient_function=lambda x, y: self.lambda_, \
                        trial_basis_function=self.basis_function, test_basis_function=self.basis_function, \
                        trial_derivative_order=[0, 1], test_derivative_order=[0, 1], \
                        Gaussian_Integral_triangular_2D_N=self.Gaussian_Integral_triangular_2D_N)
        A11 = A1 + 2*A2 + A3        
        A12 = A4 + A5
        A21 = A6 + A7
        A22 = A8 + 2*A3 + A2
        b1 = assemble_vector_b_2D(P, T, Tb, self.lambda_, self.mu, self.source1_function, self.basis_function, self.Gaussian_Integral_triangular_2D_N)
        b2 = assemble_vector_b_2D(P, T, Tb, self.lambda_, self.mu, self.source2_function, self.basis_function, self.Gaussian_Integral_triangular_2D_N)
        A = sp.bmat([[A11, A12],
                    [A21, A22]], format='lil')
        b = sp.bmat([[b1],
                    [b2]], format='lil')
        A, b = treat_Dirichlet_boundary_2D(Pb, Tb, A, b, boundary_edges, self.dirichlet_boundary1, self.dirichlet_boundary2)
        uh = scipy.sparse.linalg.spsolve(A.tocsc(), b.tocsc())
        # uh = scipy.sparse.linalg.lsqr(A.tocsc(), b.tocsc())
        return uh

