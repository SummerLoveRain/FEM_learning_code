import numpy as np
import scipy
from basis import local_linear_1D_basis, local_quadratic_1D_basis, Basis
from fem import *
from integration import integration, Gaussian_Integral_1D

class Base_1D_Solver:
    def __init__(self, a, b, number_of_elements, basis_function_type_trial, basis_function_type_test, Gaussian_Integral_1D_N=3):
        self.a = a
        self.b = b
        self.number_of_elements = number_of_elements
        self.basis_function_type_trial = basis_function_type_trial
        self.basis_function_type_test = basis_function_type_test
        self.Gaussian_Integral_1D_N = Gaussian_Integral_1D_N

        if self.basis_function_type_trial == Basis.LINEAR:
            self.basis_function_trial = local_linear_1D_basis
        elif self.basis_function_type_test == Basis.QUADRATIC:
            self.basis_function_trial = local_quadratic_1D_basis
        else:
            ValueError("The basis_function_type_trial is not correct!")

        if self.basis_function_type_test == Basis.LINEAR:
            self.basis_function_test = local_linear_1D_basis
        elif self.basis_function_type_test == Basis.QUADRATIC:
            self.basis_function_test = local_quadratic_1D_basis
        else:
            ValueError("The basis_function_type_test is not correct!")
    
    def generate_P_T(self):
        P, T = self.generate_Pb_Tb_linear()
        return P, T
    
    def generate_Pb_Tb_linear(self):
        a = self.a
        b = self.b
        number_of_elements = self.number_of_elements

        Pb = np.linspace(a, b, number_of_elements+1)
        Tb = np.zeros((number_of_elements, 2), dtype=int)
        Tb[:, 0] = np.arange(number_of_elements)
        Tb[:, 1] = Tb[:, 0] + 1
        return Pb, Tb
    
    def generate_Pb_Tb_quadratic(self):
        a = self.a
        b = self.b
        number_of_elements = self.number_of_elements

        Pb = np.linspace(a, b, 2*number_of_elements+1)
        Tb = np.zeros((number_of_elements, 3), dtype=int)
        Tb[:, 0] = np.arange(2*number_of_elements, step=2)
        Tb[:, 1] = Tb[:, 0] + 2
        Tb[:, 2] = Tb[:, 0] + 1
        return Pb, Tb
    
    def generate_Pb_Tb_trial(self):
        if self.basis_function_type_trial == Basis.LINEAR:
            Pb, Tb = self.generate_Pb_Tb_linear()
        elif self.basis_function_type_trial == Basis.QUADRATIC:
            Pb, Tb = self.generate_Pb_Tb_quadratic()
        return Pb, Tb
    
    def generate_Pb_Tb_test(self):
        if self.basis_function_type_test == Basis.LINEAR:
            Pb, Tb = self.generate_Pb_Tb_linear()
        elif self.basis_function_type_test == Basis.QUADRATIC:
            Pb, Tb = self.generate_Pb_Tb_quadratic()
        return Pb, Tb
    
    def solve_x(self, x):
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb_trial, Tb_trial = self.generate_Pb_Tb_trial()
        # N_trial:  the total number of the trial functions in the local space
        N_local_trial = Tb_trial.shape[1]
        derivative_order_trial = 0

        number_of_elements = T.shape[0]
        def func(x):
            value = 0.0
            for i in range(number_of_elements):
                # lower bound
                x_i = P[T[i, 0]]
                # upper bound
                x_ip = P[T[i, 1]]
                if x>x_i and x<x_ip:
                    for alpha in range(N_local_trial):
                        f_func = lambda x : self.uh[Tb_trial[i, alpha]]*self.basis_function_trial(x_i, x_ip, x, alpha, derivative_order_trial)
                        value += f_func(x)
                elif x==x_i:
                    value = self.uh[Tb_trial[i, 0]]
                elif x==x_ip:
                    value = self.uh[Tb_trial[i, 1]]
            return value
        func_x =  np.frompyfunc(func, 1, 1)
        value = func_x(x)
        return value
    
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
        Pb_trial, Tb_trial = self.generate_Pb_Tb_trial()
        # N_trial:  the total number of the trial functions in the local space
        N_local_trial = Tb_trial.shape[1]

        number_of_elements = T.shape[0]
        error = 0.0

        for i in range(number_of_elements):
            # lower bound
            x_i = P[T[i, 0]]
            # upper bound
            x_ip = P[T[i, 1]]
            def uh_func(x):
                uh = 0.0
                for alpha in range(N_local_trial):
                    uh += self.uh[Tb_trial[i, alpha]]*self.basis_function_trial(x_i, x_ip, x, alpha, derivative_order=0)
                return uh
            f_integral = lambda x : np.abs(u_func(x) - uh_func(x))
            # Gaussian Integral
            integral = Gaussian_Integral_1D(x_i, x_ip, f_integral, self.Gaussian_Integral_1D_N, integration_type=integration.MAX)
            error += integral
        return error
        

    def L2_error(self, u_func):
        '''
        u_func: The function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb_trial, Tb_trial = self.generate_Pb_Tb_trial()
        # N_trial:  the total number of the trial functions in the local space
        N_local_trial = Tb_trial.shape[1]

        number_of_elements = T.shape[0]
        error = 0.0

        for i in range(number_of_elements):
            # lower bound
            x_i = P[T[i, 0]]
            # upper bound
            x_ip = P[T[i, 1]]
            def uh_func(x):
                uh = 0.0
                for alpha in range(N_local_trial):
                    uh += self.uh[Tb_trial[i, alpha]]*self.basis_function_trial(x_i, x_ip, x, alpha, derivative_order=0)
                return uh
            f_integral = lambda x : (u_func(x) - uh_func(x))**2
            # Gaussian Integral
            integral = Gaussian_Integral_1D(x_i, x_ip, f_integral, self.Gaussian_Integral_1D_N)
            error += integral
        error = np.sqrt(error)
        return error
    
    def H1_error(self, u_func, u_derivative_func):
        '''
        u_func: The function of exact solution
        u_derivative_func: The derivative function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb_trial, Tb_trial = self.generate_Pb_Tb_trial()
        # N_trial:  the total number of the trial functions in the local space
        N_local_trial = Tb_trial.shape[1]

        number_of_elements = T.shape[0]
        error = 0.0
        for i in range(number_of_elements):
            # lower bound
            x_i = P[T[i, 0]]
            # upper bound
            x_ip = P[T[i, 1]]            
            def uh_func(x):
                uh = 0.0
                for alpha in range(N_local_trial):
                    uh += self.uh[Tb_trial[i, alpha]]*self.basis_function_trial(x_i, x_ip, x, alpha, derivative_order=0)
                return uh
            def uh_derivative_func(x):
                uh = 0.0
                for alpha in range(N_local_trial):
                    uh += self.uh[Tb_trial[i, alpha]]*self.basis_function_trial(x_i, x_ip, x, alpha, derivative_order=1)
                return uh            
            f_integral = lambda x : (u_func(x) - uh_func(x))**2 + (u_derivative_func(x) - uh_derivative_func(x))**2
            # Gaussian Integral
            integral = Gaussian_Integral_1D(x_i, x_ip, f_integral, self.Gaussian_Integral_1D_N)
            error += integral
        error = np.sqrt(error)
        return error
    
    
    def H1_semi_error(self, u_derivative_func):
        '''
        u_derivative_func: The derivative function of exact solution
        '''
        if self.uh is None:
            self.uh = self.solve()
        P, T = self.generate_P_T()
        Pb_trial, Tb_trial = self.generate_Pb_Tb_trial()
        # N_trial:  the total number of the trial functions in the local space
        N_local_trial = Tb_trial.shape[1]

        number_of_elements = T.shape[0]
        error = 0.0
        for i in range(number_of_elements):
            # lower bound
            x_i = P[T[i, 0]]
            # upper bound
            x_ip = P[T[i, 1]]
            def uh_derivative_func(x):
                uh = 0.0
                for alpha in range(N_local_trial):
                    uh += self.uh[Tb_trial[i, alpha]]*self.basis_function_trial(x_i, x_ip, x, alpha, derivative_order=1)
                return uh            
            f_integral = lambda x : (u_derivative_func(x) - uh_derivative_func(x))**2
            # Gaussian Integral
            integral = Gaussian_Integral_1D(x_i, x_ip, f_integral, self.Gaussian_Integral_1D_N)
            error += integral
        error = np.sqrt(error)
        return error

class Possion_1D_FE_solver_Dirichlet(Base_1D_Solver):
    def __init__(self, a, b, coefficient_function, source_function, number_of_elements, basis_function_type_trial, basis_function_type_test, Gaussian_Integral_1D_N=3):
        '''
        a : lower bound
        b : upper bound
        basis_function_type: linear, quadratic
        '''
        super(Possion_1D_FE_solver_Dirichlet, self).__init__(a, b, number_of_elements, basis_function_type_trial, basis_function_type_test, Gaussian_Integral_1D_N)
        self.coefficient_function = coefficient_function
        self.source_function = source_function
        
        self.uh = None
    
    
    def dirichlet_boundary(self, x):
        a = self.a
        b = self.b
        if np.abs(x-a)<1e-15:
            result = 0.0
        elif np.abs(x-b)<1e-15:
            result = np.cos(b)
        else:
            ValueError('No boundary points!')
        return result
    
    def generate_boundary_nodes(self):
        #boundarynodes[i,0] 表示第i个边界点的边界类型:
        #0-- dirichlet, 1--neumann, 2--robin
        #boundarynodes[i,1] 表示第i个边界点的全局坐标
        boundarynodes = np.zeros((2,2),dtype=int)

        boundarynodes[0,0] = Boundary.DIRICHLET_BOUNDARY
        boundarynodes[0,1] = 0

        boundarynodes[1,0] = Boundary.DIRICHLET_BOUNDARY

        number_of_elements = self.number_of_elements
        if self.basis_function_type_trial == Basis.LINEAR:
            boundarynodes[1,1] = number_of_elements
        elif self.basis_function_type_trial == Basis.QUADRATIC:
            boundarynodes[1,1] = 2*number_of_elements
        return boundarynodes
    
    def solve(self):
        P, T = self.generate_P_T()
        Pb_trial, Tb_trial = self.generate_Pb_Tb_trial()
        Pb_test, Tb_test = self.generate_Pb_Tb_test()
        boundary_nodes = self.generate_boundary_nodes()
        derivative_order_trial = 1
        derivative_order_test = 1
        A = assemble_matrix_A_1D(P, T, Tb_trial, Tb_test, self.coefficient_function, self.basis_function_trial, self.basis_function_test, \
                        derivative_order_trial, derivative_order_test, Gaussian_Integral_1D_N=self.Gaussian_Integral_1D_N)
        b = assemble_vector_b_1D(P, T, Tb_test, self.source_function, self.basis_function_test, Gaussian_Integral_1D_N=self.Gaussian_Integral_1D_N)
        A, b = treat_Dirichlet_boundary_1D(Pb_trial, A, b, boundary_nodes, self.dirichlet_boundary)
        uh = scipy.sparse.linalg.spsolve(A.tocsc(), b.tocsc())
        # uh = scipy.sparse.linalg.lsqr(A.tocsc(), b.tocsc())
        return uh



class Possion_1D_FE_solver_Neumann(Base_1D_Solver):
    def __init__(self, a, b, coefficient_function, source_function, number_of_elements, basis_function_type_trial, basis_function_type_test, Gaussian_Integral_1D_N=3):
        '''
        a : lower bound
        b : upper bound
        basis_function_type: linear, quadratic
        '''
        super(Possion_1D_FE_solver_Neumann, self).__init__(a, b, number_of_elements, basis_function_type_trial, basis_function_type_test, Gaussian_Integral_1D_N)
        self.coefficient_function = coefficient_function
        self.source_function = source_function

        self.uh = None
    
    def dirichlet_boundary(self, x):
        a = self.a
        b = self.b
        if np.abs(x-a)<1e-15:
            result = 0.0
        # elif np.abs(x-b)<1e-15:
        #     result = np.cos(b)
        else:
            ValueError('No boundary points!')
        return result
    
    def neumann_boundary(self, x):
        b = self.b
        if np.abs(x-b)<1e-15:
            result = np.cos(x) - np.sin(x)
        else:
            ValueError('No boundary points!')
        return result
    
    def generate_boundary_nodes(self):
        #boundarynodes[i,0] 表示第i个边界点的边界类型:
        #0-- dirichlet, 1--neumann, 2--robin
        #boundarynodes[i,1] 表示第i个边界点的全局坐标
        boundarynodes = np.zeros((2,3),dtype=int)

        boundarynodes[0,0] = Boundary.DIRICHLET_BOUNDARY
        boundarynodes[0,1] = 0
        # normal direction
        boundarynodes[0,2] = 0

        boundarynodes[1,0] = Boundary.NEUMANN_BOUNDARY
        # normal direction
        boundarynodes[1,2] = 1

        number_of_elements = self.number_of_elements
        if self.basis_function_type_trial == Basis.LINEAR:
            boundarynodes[1,1] = number_of_elements
        if self.basis_function_type_trial == Basis.QUADRATIC:
            boundarynodes[1,1] = 2*number_of_elements
        return boundarynodes
    
    def solve(self):
        P, T = self.generate_P_T()
        Pb_trial, Tb_trial = self.generate_Pb_Tb_trial()
        Pb_test, Tb_test = self.generate_Pb_Tb_test()
        boundary_nodes = self.generate_boundary_nodes()
        derivative_order_trial = 1
        derivative_order_test = 1
        A = assemble_matrix_A_1D(P, T, Tb_trial, Tb_test, self.coefficient_function, self.basis_function_trial, self.basis_function_test, \
                        derivative_order_trial, derivative_order_test, Gaussian_Integral_1D_N=self.Gaussian_Integral_1D_N)
        b = assemble_vector_b_1D(P, T, Tb_test, self.source_function, self.basis_function_test, Gaussian_Integral_1D_N=self.Gaussian_Integral_1D_N)
        A, b = treat_Neumann_boundary_1D(Pb_trial, Tb_test, A, b, boundary_nodes, self.coefficient_function, self.neumann_boundary, self.basis_function_test)
        A, b = treat_Dirichlet_boundary_1D(Pb_trial, A, b, boundary_nodes, self.dirichlet_boundary)
        uh = scipy.sparse.linalg.spsolve(A.tocsc(), b.tocsc())
        # uh = scipy.sparse.linalg.lsqr(A.tocsc(), b.tocsc())
        return uh



class Possion_1D_FE_solver_Robin(Base_1D_Solver):
    def __init__(self, a, b, coefficient_function, source_function, number_of_elements, basis_function_type_trial, basis_function_type_test, Gaussian_Integral_1D_N=3):
        '''
        a : lower bound
        b : upper bound
        basis_function_type: linear, quadratic
        '''
        super(Possion_1D_FE_solver_Robin, self).__init__(a, b, number_of_elements, basis_function_type_trial, basis_function_type_test, Gaussian_Integral_1D_N)
        self.coefficient_function = coefficient_function
        self.source_function = source_function
        
        self.uh = None
    
    def dirichlet_boundary(self, x):
        b = self.b
        if np.abs(x-b)<1e-15:
            result = np.cos(x)
        else:
            ValueError('No boundary points!')
        return result
    
    def robin_boundary_p(self, x):
        a = self.a
        if np.abs(x-a)<1e-15:
            result = 1.0
        else:
            ValueError('No boundary points!')
        return result
    
    def robin_boundary_q(self, x):
        a = self.a
        if np.abs(x-a)<1e-15:
            result = 1.0
        else:
            ValueError('No boundary points!')
        return result
    
    def generate_boundary_nodes(self):
        #boundarynodes[i,0] 表示第i个边界点的边界类型:
        #0-- dirichlet, 1--neumann, 2--robin
        #boundarynodes[i,1] 表示第i个边界点的全局坐标
        boundarynodes = np.zeros((2,3),dtype=int)

        boundarynodes[0,0] = Boundary.ROBIN_BOUNDARY
        boundarynodes[0,1] = 0
        # normal direction
        boundarynodes[0,2] = -1

        boundarynodes[1,0] = Boundary.DIRICHLET_BOUNDARY
        boundarynodes[1,2] = 0

        number_of_elements = self.number_of_elements
        if self.basis_function_type_trial == Basis.LINEAR:
            boundarynodes[1,1] = number_of_elements
        if self.basis_function_type_trial == Basis.QUADRATIC:
            boundarynodes[1,1] = 2*number_of_elements
        return boundarynodes
    
    def solve(self):
        P, T = self.generate_P_T()
        Pb_trial, Tb_trial = self.generate_Pb_Tb_trial()
        Pb_test, Tb_test = self.generate_Pb_Tb_test()
        boundary_nodes = self.generate_boundary_nodes()
        derivative_order_trial = 1
        derivative_order_test = 1
        A = assemble_matrix_A_1D(P, T, Tb_trial, Tb_test, self.coefficient_function, self.basis_function_trial, self.basis_function_test, \
                        derivative_order_trial, derivative_order_test, Gaussian_Integral_1D_N=self.Gaussian_Integral_1D_N)
        b = assemble_vector_b_1D(P, T, Tb_test, self.source_function, self.basis_function_test, Gaussian_Integral_1D_N=self.Gaussian_Integral_1D_N)
        A, b = treat_Robin_boundary_1D(Pb_trial, Tb_trial, Tb_test, A, b, boundary_nodes, self.coefficient_function, \
                                       self.robin_boundary_p, self.robin_boundary_q, self.basis_function_trial, self.basis_function_test)
        A, b = treat_Dirichlet_boundary_1D(Pb_trial, A, b, boundary_nodes, self.dirichlet_boundary)
        uh = scipy.sparse.linalg.spsolve(A.tocsc(), b.tocsc())
        # uh = scipy.sparse.linalg.lsqr(A.tocsc(), b.tocsc())
        return uh
