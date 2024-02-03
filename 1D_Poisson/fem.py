import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from integration import Gaussian_Integral_1D
from basis import affine_x

class Boundary:
    DIRICHLET_BOUNDARY = 0
    NEUMANN_BOUNDARY = 1
    ROBIN_BOUNDARY = 2

def assemble_matrix_A_1D(P, T, Tb_trial, Tb_test, coefficient_function, basis_function_trial, basis_function_test, \
                         derivative_order_trial, derivative_order_test, Gaussian_Integral_1D_N=3):
    '''
    P: Information matrix to store the coordinates of the mesh node
    T: Information matrix to store the global node indices of the mesh nodes of the mesh element
    Tb_trial: Information matrix for the finite element nodes corresponding to the basis functions in the trial space
    Tb_test: Information matrix for the finite element nodes corresponding to the basis functions in the test space
    coefficient_function: coefficient function in the integral
    basis_function_trial: The basis functions in the trial space
    basis_function_test: The basis functions in the test space
    derivative_order_trial: The derivative order of the trial function
    derivative_order_test: The derivative order of the test function
    Gaussian_Integral_1D_N: The number of points which is used in Gaussian Integral and the default value is 3 
    return A: stiffness matrix A
    '''

    # N_trial:  the total number of the finite element basis functions in the trial space
    N_global_trial = np.max(Tb_trial) + 1
    # N_test:  the total number of the finite element basis functions in the test space
    N_globa_test = np.max(Tb_test) + 1
    
    # N_trial:  the total number of the trial functions in the local space
    N_local_trial = Tb_trial.shape[1]
    # N_test:  the total number of the test functions in the local space
    N_local_test  = Tb_test.shape[1]

    # define a sparse matrix
    A = lil_matrix((N_globa_test, N_global_trial))
    number_of_elements = T.shape[0]
    for i in range(number_of_elements):
        for alpha in range(N_local_trial):
            for beta in range(N_local_test):
                # lower bound
                x_i = P[T[i, 0]]
                # upper bound
                x_ip = P[T[i, 1]]
                f_integral = lambda x : coefficient_function(x) * \
                        basis_function_trial(affine_x(x_i, x_ip, x, derivative_order=0), alpha, derivative_order_trial) * \
                        basis_function_test(affine_x(x_i, x_ip, x, derivative_order=0), beta, derivative_order_test) * \
                        affine_x(x_i, x_ip, x, derivative_order=derivative_order_trial) * \
                        affine_x(x_i, x_ip, x, derivative_order=derivative_order_test)
                # Gaussian Integral
                integral = Gaussian_Integral_1D(x_i, x_ip, f_integral, Gaussian_Integral_1D_N)
                A[Tb_test[i, beta], Tb_trial[i, alpha]] += integral
    return A

def assemble_vector_b_1D(P, T, Tb_test, source_function, basis_function_test, Gaussian_Integral_1D_N=3):
    '''
    P: Information matrix to store the coordinates of the mesh node
    T: Information matrix to store the global node indices of the mesh nodes of the mesh element
    Tb_test: Information matrix for the finite element nodes corresponding to the basis functions in the test space
    basis_function_test: The basis functions in the test space
    Gaussian_Integral_1D_N: The number of points which is used in Gaussian Integral and the default value is 3 
    return b: load vector b
    '''
    # N_test:  the total number of the finite element basis functions in the test space
    N_globa_test = np.max(Tb_test) + 1
    # N_test:  the total number of the test functions in the local space
    N_local_test  = Tb_test.shape[1]
    # define a sparse matrix
    b = lil_matrix((N_globa_test, 1))
    number_of_elements = T.shape[0]
    for i in range(number_of_elements):
        for beta in range(N_local_test):
            # lower bound
            x_i = P[T[i, 0]]
            # upper bound
            x_ip = P[T[i, 1]]
            f_integral = lambda x : source_function(x) * \
                    basis_function_test(affine_x(x_i, x_ip, x, derivative_order=0), beta, 0)
            # Gaussian Integral
            integral = Gaussian_Integral_1D(x_i, x_ip, f_integral, Gaussian_Integral_1D_N)
            b[Tb_test[i, beta], 0] += integral
    return b

def treat_Dirichlet_boundary_1D(Pb_trial, A, b, boundary_nodes, boundary_function):
    '''
    Pb_trial: Information matrix to store the coordinates of the mesh node used in the trial function
    A: stiffness matrix
    b: load vector
    boundary_nodes: nodes on the boundary
    boundary_function: function on the boundary
    '''
    #boundarynodes[i,0] 表示第i个边界点的边界类型:
    #0-- dirichlet, 1--neumann, 2--robin
    #boundarynodes[i,1] 表示第i个边界点的全局坐标
    nodes_N = boundary_nodes.shape[0]
    for i in range(nodes_N):
        if boundary_nodes[i, 0] == Boundary.DIRICHLET_BOUNDARY:
            j = boundary_nodes[i, 1]
            A[j] = 0
            A[j, j] = 1
            b[j, 0] = boundary_function(Pb_trial[j])
    return A, b


def treat_Neumann_boundary_1D(Pb_trial, Tb_test, A, b, boundary_nodes, coefficient_function, boundary_function, basis_function_test):
    '''
    Pb_trial: Information matrix to store the coordinates of the mesh node used in the trial function
    A: stiffness matrix
    b: load vector
    boundary_nodes: nodes on the boundary
    boundary_function: function on the boundary
    '''
    #boundarynodes[i,0] 表示第i个边界点的边界类型:
    #0-- dirichlet, 1--neumann, 2--robin

    # N_test:  the total number of the test functions in the local space
    N_local_test = Tb_test.shape[1]
    #boundarynodes[i,1] 表示第i个边界点的全局坐标
    nodes_N = boundary_nodes.shape[0]
    for i in range(nodes_N):
        if boundary_nodes[i, 0] == Boundary.NEUMANN_BOUNDARY:
            j = boundary_nodes[i, 1]
            normal_direction = boundary_nodes[i, 2]
            if j == 0:
                x_i = Pb_trial[j]
                x_ip = Pb_trial[j+N_local_test-1]
            else:
                # lower bound
                x_i = Pb_trial[j-N_local_test+1]
                # upper bound
                x_ip = Pb_trial[j]
            for beta in range(N_local_test):
                b[j, 0] += normal_direction*coefficient_function(Pb_trial[j]) * boundary_function(Pb_trial[j]) * \
                        basis_function_test(affine_x(x_i, x_ip, Pb_trial[j], derivative_order=0), beta, 0)
    return A, b


def treat_Robin_boundary_1D(Pb_trial, Tb_trial, Tb_test, A, b, boundary_nodes, coefficient_function, boundary_function_p, boundary_function_q, \
                            basis_function_trial, basis_function_test):
    '''
    Pb_trial: Information matrix to store the coordinates of the mesh node used in the trial function
    A: stiffness matrix
    b: load vector
    boundary_nodes: nodes on the boundary
    boundary_function: function on the boundary
    '''
    #boundarynodes[i,0] 表示第i个边界点的边界类型:
    #0-- dirichlet, 1--neumann, 2--robin

    # N_trial:  the total number of the trial functions in the local space
    N_local_trial = Tb_trial.shape[1]
    # N_test:  the total number of the test functions in the local space
    N_local_test = Tb_test.shape[1]
    #boundarynodes[i,1] 表示第i个边界点的全局坐标
    nodes_N = boundary_nodes.shape[0]
    
    for i in range(nodes_N):
        if boundary_nodes[i, 0] == Boundary.ROBIN_BOUNDARY:
            j = boundary_nodes[i, 1]  
            normal_direction = boundary_nodes[i, 2]          
            if j == 0:
                x_i = Pb_trial[j]
                x_ip = Pb_trial[j+N_local_test-1]
            else:
                # lower bound
                x_i = Pb_trial[j-N_local_test+1]
                # upper bound
                x_ip = Pb_trial[j]
            for alpha in range(N_local_trial):
                for beta in range(N_local_test):
                    A[Tb_test[j, beta], Tb_trial[j, alpha]] += normal_direction * coefficient_function(Pb_trial[j]) * boundary_function_q(Pb_trial[j]) * \
                        basis_function_trial(affine_x(x_i, x_ip, Pb_trial[j], derivative_order=0), alpha, 0) * \
                        basis_function_test(affine_x(x_i, x_ip, Pb_trial[j], derivative_order=0), beta, 0)
            for beta in range(N_local_test):
                b[j, 0] += normal_direction*coefficient_function(Pb_trial[j]) * boundary_function_p(Pb_trial[j]) * \
                        basis_function_test(affine_x(x_i, x_ip, Pb_trial[j], derivative_order=0), beta, 0)
    return A, b