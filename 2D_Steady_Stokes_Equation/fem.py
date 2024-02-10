import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from integration import Gaussian_Integral_1D, Gaussian_Integral_triangular_2D
from basis import affine_triangular_xy

class Boundary:
    DIRICHLET_BOUNDARY = 0
    NEUMANN_BOUNDARY = 1
    ROBIN_BOUNDARY = 2

def integrate_trial_test_basis_function_2D(P, T, Pb_trial, Tb_trial, Pb_test, Tb_test, coefficient_function, \
                                        trial_basis_function, test_basis_function, \
                                        trial_derivative_order, test_derivative_order, Gaussian_Integral_triangular_2D_N):
    # the number of local basis functions
    Nl_trial = Tb_trial.shape[1]
    Nl_test = Tb_test.shape[1]
    # the number of finite elements
    Nb_trial = Tb_trial.shape[0]
    Nb_test = Tb_test.shape[0]
    if Nb_trial!=Nb_test:
        ValueError("The number of finite elments is not correct!")
    # the number of nodes
    Ng_trial = np.max(Tb_trial) + 1
    Ng_test = np.max(Tb_test) + 1
    # define a sparse matrix
    A = lil_matrix((Ng_test, Ng_trial))
    for i in range(Nb_trial):
        vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
        for j in range(Nl_test):
            for k in range(Nl_trial):
                def f_integral(x, y):
                    coe = coefficient_function(x, y)
                    u = trial_basis_function(x, y, vertices, k, derivative_order_x=trial_derivative_order[0], derivative_order_y=trial_derivative_order[1])
                    v = test_basis_function(x, y, vertices, j, derivative_order_x=test_derivative_order[0], derivative_order_y=test_derivative_order[1])
                    result = coe * u * v
                    return result
                # Gaussian Integral
                integral = Gaussian_Integral_triangular_2D(vertices, f_integral, Gaussian_Integral_triangular_2D_N)
                A[Tb_test[i, j], Tb_trial[i, k]] += integral
    return A

def assemble_vector_b_2D(P, T, Tb, nu, source_function, basis_function, Gaussian_Integral_triangular_2D_N=10):
    # the number of local basis functions
    Nl = Tb.shape[1]
    # the number of finite elements
    Nb = T.shape[0]
    # the number of nodes
    Ng = np.max(Tb) + 1
    # define a sparse matrix
    b = lil_matrix((Ng, 1))
    for i in range(Nb):
        vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
        for j in range(Nl):
            def f_integral(x, y): 
                result = source_function(x, y, nu) * basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                return result
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, Gaussian_Integral_triangular_2D_N)
            b[Tb[i, j], 0] += integral
    return b

def treat_Dirichlet_boundary_2D(Pb, Tb, A, b, boundary_edges, boundary_function1, boundary_function2):
    # the number of boundary node on edge ek
    B_size = boundary_edges.shape[1]
    # the number of boundary nodes
    N_nodes = boundary_edges.shape[0]
    # the number of nodes
    Ng = np.max(Tb) + 1
    for i in range(N_nodes):
        if boundary_edges[i, 0] == Boundary.DIRICHLET_BOUNDARY:
            for j in range(2, B_size):
                index = boundary_edges[i, j]
                x = Pb[index, 0]
                y = Pb[index, 1]
                
                A[index] = 0
                A[index, index] = 1
                b[index, 0] = boundary_function1(x, y)

                A[index+Ng] = 0
                A[index+Ng, index+Ng] = 1
                b[index+Ng, 0] = boundary_function2(x, y)
    return A, b


def treat_Dirichlet_boundary_2D2(P, T, Pb, Tb, A, b, boundary_edges, boundary_function1, boundary_function2, boundary_edges2, boundary_function3):
    # the number of boundary node on edge ek
    B_size = boundary_edges.shape[1]
    # the number of boundary nodes
    N_nodes = boundary_edges.shape[0]
    # the number of nodes
    Ng = np.max(Tb) + 1
    for i in range(N_nodes):
        if boundary_edges[i, 0] == Boundary.DIRICHLET_BOUNDARY:
            for j in range(2, B_size):
                index = boundary_edges[i, j]
                x = Pb[index, 0]
                y = Pb[index, 1]
                
                A[index] = 0
                A[index, index] = 1
                b[index, 0] = boundary_function1(x, y)

                A[index+Ng] = 0
                A[index+Ng, index+Ng] = 1
                b[index+Ng, 0] = boundary_function2(x, y)

    # the number of boundary node on edge ek
    B_size = boundary_edges2.shape[1]
    # the number of boundary nodes
    N_nodes = boundary_edges2.shape[0]
    for i in range(N_nodes):
        if boundary_edges2[i, 0] == Boundary.DIRICHLET_BOUNDARY:
            for j in range(2, B_size):
                index = boundary_edges2[i, j]
                x = P[index, 0]
                y = P[index, 1]
                A[index+2*Ng] = 0
                A[index+2*Ng, index+2*Ng] = 1
                b[index+2*Ng, 0] = boundary_function3(x, y)
    return A, b