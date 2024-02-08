import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from integration import Gaussian_Integral_1D, Gaussian_Integral_triangular_2D
from basis import affine_triangular_xy

class Boundary:
    DIRICHLET_BOUNDARY = 0
    NEUMANN_BOUNDARY = 1
    ROBIN_BOUNDARY = 2

    
def assemble_matrix_M_2D(P, T, Pb, Tb, basis_function, Gaussian_Integral_triangular_2D_N=10):
    # the number of local basis functions
    Nl = Tb.shape[1]
    # the number of finite elements
    Nb = Tb.shape[0]
    # the number of nodes
    Ng = np.max(Tb) + 1
    # define a sparse matrix
    M = lil_matrix((Ng, Ng))
    for i in range(Nb):
        vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
        for j in range(Nl):
            for k in range(Nl):
                def f_integral(x, y):
                    new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                    u = basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=0)
                    v = basis_function(new_x, new_y, k, derivative_order_x=0, derivative_order_y=0)
                    result = u*v
                    return result
                # Gaussian Integral
                integral = Gaussian_Integral_triangular_2D(vertices, f_integral, Gaussian_Integral_triangular_2D_N)
                M[Tb[i, k], Tb[i, j]] += integral
    return M

def assemble_matrix_A_2D(P, T, Pb, Tb, coefficient_function, basis_function, Gaussian_Integral_triangular_2D_N=10):
    # the number of local basis functions
    Nl = Tb.shape[1]
    # the number of finite elements
    Nb = Tb.shape[0]
    # the number of nodes
    Ng = np.max(Tb) + 1
    # define a sparse matrix
    A = lil_matrix((Ng, Ng))
    for i in range(Nb):
        vertices = np.array([[P[T[i, 0], 0], P[T[i, 0], 1]], [P[T[i, 1], 0], P[T[i, 1], 1]], [P[T[i, 2], 0], P[T[i, 2], 1]]])
        for j in range(Nl):
            for k in range(Nl):
                def f_integral(x, y):
                    coe = coefficient_function(x, y)
                    new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                    ux = basis_function(new_x, new_y, j, derivative_order_x=1, derivative_order_y=0)
                    uy = basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=1)
                    vx = basis_function(new_x, new_y, k, derivative_order_x=1, derivative_order_y=0)
                    vy = basis_function(new_x, new_y, k, derivative_order_x=0, derivative_order_y=1)
                    [xhat_x, yhat_x] = affine_triangular_xy(x, y, vertices, derivative_order_x=1, derivative_order_y=0)
                    [xhat_y, yhat_y] = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=1)
                    result = coe* ((ux*xhat_x + uy*yhat_x)*(vx*xhat_x + vy*yhat_x) + (ux*xhat_y + uy*yhat_y)*(vx*xhat_y + vy*yhat_y))
                    return result
                # Gaussian Integral
                integral = Gaussian_Integral_triangular_2D(vertices, f_integral, Gaussian_Integral_triangular_2D_N)
                A[Tb[i, k], Tb[i, j]] += integral
    return A

def assemble_vector_b_2D(P, T, Tb, t, source_function, basis_function, Gaussian_Integral_triangular_2D_N=10):
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
                new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
                result = source_function(x, y, t) * basis_function(new_x, new_y, j, derivative_order_x=0, derivative_order_y=0)
                return result
            # Gaussian Integral
            integral = Gaussian_Integral_triangular_2D(vertices, f_integral, Gaussian_Integral_triangular_2D_N)
            b[Tb[i, j], 0] += integral
    return b

def treat_Dirichlet_boundary_2D(Pb, Tb, A, b, boundary_edges, boundary_function):
    # the number of boundary node on edge ek
    B_size = boundary_edges.shape[1]
    # the number of boundary nodes
    N_nodes = boundary_edges.shape[0]
    for i in range(N_nodes):
        if boundary_edges[i, 0] == Boundary.DIRICHLET_BOUNDARY:
            for j in range(2, B_size):
                index = boundary_edges[i, j]
                x = Pb[index, 0]
                y = Pb[index, 1]
                A[index] = 0
                A[index, index] = 1
                b[index, 0] = boundary_function(x, y)
    return A, b

def treat_Dirichlet_boundary_2D_A(A, boundary_edges):
    # the number of boundary node on edge ek
    B_size = boundary_edges.shape[1]
    # the number of boundary nodes
    N_nodes = boundary_edges.shape[0]
    for i in range(N_nodes):
        if boundary_edges[i, 0] == Boundary.DIRICHLET_BOUNDARY:
            for j in range(2, B_size):
                index = boundary_edges[i, j]
                A[index] = 0
                A[index, index] = 1
    return A


def treat_Dirichlet_boundary_2D_b(Pb, Tb, b, t, boundary_edges, boundary_function):
    # the number of boundary node on edge ek
    B_size = boundary_edges.shape[1]
    # the number of boundary nodes
    N_nodes = boundary_edges.shape[0]
    for i in range(N_nodes):
        if boundary_edges[i, 0] == Boundary.DIRICHLET_BOUNDARY:
            for j in range(2, B_size):
                index = boundary_edges[i, j]
                x = Pb[index, 0]
                y = Pb[index, 1]
                b[index, 0] = boundary_function(x, y, t)
    return b

