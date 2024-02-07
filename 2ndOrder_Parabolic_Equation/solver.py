import math
import numpy as np
import scipy
from basis import basis_triangular_reference_linear_2D, basis_triangular_reference_quadratic_2D, Basis
from fem import *
from integration import integration, Gaussian_Integral_triangular_2D

class Base_2D_Solver:
    def __init__(self, bounds, Nx, Ny, Nt, theta, basis_function_type, Gaussian_Integral_triangular_2D_N=10):
        # the bounds of problems in a rectangular domain
        self.bounds = bounds        
        self.a = self.bounds[0]
        self.b = self.bounds[1]
        self.c = self.bounds[2]
        self.d = self.bounds[3]
        self.initial_time = self.bounds[4]
        self.end_time = self.bounds[5]
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
        self.basis_function_type = basis_function_type
        # the order used in Gaussian Integrals
        self.Gaussian_Integral_triangular_2D_N = Gaussian_Integral_triangular_2D_N

        if self.basis_function_type == Basis.LINEAR:
            self.basis_function = basis_triangular_reference_linear_2D
        elif self.basis_function_type == Basis.QUADRATIC:
            self.basis_function = basis_triangular_reference_quadratic_2D
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

    def L_infinity_error(self, u_func):
        '''
        u_func: The function of exact solution
        compute the L-infinity error
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb, Tb = self.generate_Pb_Tb()
        Nl = Tb.shape[1]

        Nb = T.shape[0]
        error=0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def uh_func(x, y):
                uh = 0.0
                new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                for j in range(Nl):
                    uh += self.uh[Tb[i, j]]*self.basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=0)
                return uh
            f_integral = lambda x ,y: np.abs(u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=0) - uh_func(x, y))
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N, integration_type=integration.MAX)
            if integral>error:
                error = integral
        return error
        

    def L2_error(self, u_func):
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
        error = 0.0

        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def uh_func(x, y):
                uh = 0.0
                new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                for j in range(Nl):
                    uh += self.uh[Tb[i, j]]*self.basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=0)
                return uh
            f_integral = lambda x ,y: (u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=0) - uh_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error
    
    def H1_error(self, u_func):
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
        error = 0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def uh_func(x, y):
                uh = 0.0
                new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                for j in range(Nl):
                    uh += self.uh[Tb[i, j]]*self.basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=0)
                return uh
            def uh_x_func(x, y):
                uh = 0.0
                new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                [xhat_x, yhat_x] = affine_triangular_xy(x, y, vertices, derivative_order_x=1, derivative_order_y=0)
                for j in range(Nl):
                    ux = self.basis_function(new_x, new_y, j, derivative_order_x=1, derivative_order_y=0)
                    uy = self.basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=1)
                    uh += self.uh[Tb[i, j]]*(ux*xhat_x + uy*yhat_x)
                return uh
            def uh_y_func(x, y):
                uh = 0.0
                new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                [xhat_y, yhat_y] = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=1)
                for j in range(Nl):
                    ux = self.basis_function(new_x, new_y, j, derivative_order_x=1, derivative_order_y=0)
                    uy = self.basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=1)
                    uh += self.uh[Tb[i, j]]*(ux*xhat_y + uy*yhat_y)
                return uh
            f_integral = lambda x, y : (u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=0) - uh_func(x, y))**2 +\
                  (u_func(x, y, self.end_time, derivative_order_x=1, derivative_order_y=0) - uh_x_func(x, y))**2 +\
                  (u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=1) - uh_y_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error
    
    
    def H1_semi_error(self, u_func):
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
        error = 0.0
        for i in range(Nb):
            vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
            def uh_x_func(x, y):
                uh = 0.0
                new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                [xhat_x, yhat_x] = affine_triangular_xy(x, y, vertices, derivative_order_x=1, derivative_order_y=0)
                for j in range(Nl):
                    ux = self.basis_function(new_x, new_y, j, derivative_order_x=1, derivative_order_y=0)
                    uy = self.basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=1)
                    uh += self.uh[Tb[i, j]]*(ux*xhat_x + uy*yhat_x)
                return uh
            def uh_y_func(x, y):
                uh = 0.0
                new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                [xhat_y, yhat_y] = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=1)
                for j in range(Nl):
                    ux = self.basis_function(new_x, new_y, j, derivative_order_x=1, derivative_order_y=0)
                    uy = self.basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=1)
                    uh += self.uh[Tb[i, j]]*(ux*xhat_y + uy*yhat_y)
                return uh
            f_integral = lambda x, y : (u_func(x, y, self.end_time, derivative_order_x=1, derivative_order_y=0) - uh_x_func(x, y))**2 +\
                  (u_func(x, y, self.end_time, derivative_order_x=0, derivative_order_y=1) - uh_y_func(x, y))**2
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, self.Gaussian_Integral_triangular_2D_N)
            error += integral
        error = np.sqrt(error)
        return error



class Parabolic_2ndOrder_FE_solver_Dirichlet(Base_2D_Solver):
    def __init__(self, bounds, Nx, Ny, Nt, theta, basis_function_type, coefficient_function, source_function, Gaussian_Integral_triangular_2D_N=10):
        '''
        bounds: the bounds of problems in a rectangular domain
        Nx, Ny: the number of partition on each axis
        the order used in Gaussian Integrals
        basis_function_type: linear, quadratic
        '''
        super(Parabolic_2ndOrder_FE_solver_Dirichlet, self).__init__(bounds, Nx, Ny, Nt, theta, basis_function_type, Gaussian_Integral_triangular_2D_N)
        self.coefficient_function = coefficient_function
        self.source_function = source_function
        
        self.uh = None

    def initial_condition(self, x, y):
        # the value on the boundary at initial time
        u = np.exp(x+y)
        return u
    
    def dirichlet_boundary(self, x, y, t):
        if np.abs(x-self.a)<1e-15:
            u = np.exp(self.a+y+t)
        elif np.abs(x-self.b)<1e-15:
            u = np.exp(self.b+y+t)
        elif np.abs(y-self.c)<1e-15:
            u = np.exp(x+self.c+t)
        elif np.abs(y-self.d)<1e-15:
            u = np.exp(x+self.d+t)
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
        A = assemble_matrix_A_2D(P, T, Pb, Tb, self.coefficient_function, self.basis_function, self.Gaussian_Integral_triangular_2D_N)
        M = assemble_matrix_M_2D(P, T, Pb, Tb, self.basis_function, self.Gaussian_Integral_triangular_2D_N)
        A_tilde = M/self.dt + self.theta*A
        A_tilde = treat_Dirichlet_boundary_2D_A(A_tilde, boundary_edges)
        x = Pb[:, 0:1]
        y = Pb[:, 1:2]
        u_t = self.initial_condition(x, y)

        for i in range(self.Nt):
            current_time = self.initial_time + (i+1)*self.dt
            b_t = assemble_vector_b_2D(P, T, Tb, current_time-self.dt, self.source_function, self.basis_function, self.Gaussian_Integral_triangular_2D_N)
            b_tp = assemble_vector_b_2D(P, T, Tb, current_time, self.source_function, self.basis_function, self.Gaussian_Integral_triangular_2D_N)
            b_tilde = self.theta*b_tp + (1-self.theta)*b_t + (M/self.dt - (1-self.theta)*A).dot(u_t)
            b_tilde = treat_Dirichlet_boundary_2D_b(Pb, Tb, b_tilde, current_time, boundary_edges, self.dirichlet_boundary)
            u_tp = scipy.sparse.linalg.spsolve(A_tilde.tocsc(), b_tilde)
            u_tp = np.reshape(u_tp, u_t.shape)
            u_t = u_tp
        uh = u_tp
        return uh

