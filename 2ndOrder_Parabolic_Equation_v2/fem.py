import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from integration import Gaussian_Integral_1D, Gaussian_Integral_triangular_2D

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
                    u = basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
                    v = basis_function(x, y, vertices, k, derivative_order_x=0, derivative_order_y=0)
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
                    ux = basis_function(x, y, vertices, j, derivative_order_x=1, derivative_order_y=0)
                    uy = basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=1)
                    vx = basis_function(x, y, vertices, k, derivative_order_x=1, derivative_order_y=0)
                    vy = basis_function(x, y, vertices, k, derivative_order_x=0, derivative_order_y=1)
                    result = coe* (ux*vx + uy*vy)
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
                result = source_function(x, y, t) * basis_function(x, y, vertices, j, derivative_order_x=0, derivative_order_y=0)
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


def treat_Neumann_boundary_2D(P, T, Pb, Tb, A, b, boundary_edges, coefficient_function, boundary_function, basis_function, Gaussian_Integral_1D_N):
    # the number of local basis functions
    Nl = Tb.shape[1]
    # the number of finite elements
    Nb = T.shape[0]
    # the number of boundary nodes
    N_nodes = boundary_edges.shape[0]
    for i in range(N_nodes):
        if boundary_edges[i, 0] == Boundary.NEUMANN_BOUNDARY:
            j = boundary_edges[i, 1]
            index1 = boundary_edges[i, 2]
            index2 = boundary_edges[i, 3]        
            x1 = Pb[index1, 0]
            y1 = Pb[index1, 1]
            x2 = Pb[index2, 0]
            y2 = Pb[index2, 1]
            vertices = np.array([[P[T[j, 0], 0], P[T[j, 0], 1]], [P[T[j, 1], 0], P[T[j, 1], 1]], [P[T[j, 2], 0], P[T[j, 2], 1]]])
            for k in range(Nl):                    
                if x1==x2:
                    if y1<=y2:
                        lower_bound = y1
                        upper_bound = y2
                    else:
                        lower_bound = y2
                        upper_bound = y1
                    def f_integral(y): 
                        x = x1
                        result = coefficient_function(x, y) * boundary_function(x, y) * basis_function(x, y, vertices, k, derivative_order_x=0, derivative_order_y=0)
                        return result
                elif y1==y2:
                    if x1<=x2:
                        lower_bound = x1
                        upper_bound = x2
                    else:
                        lower_bound = x2
                        upper_bound = x1
                    def f_integral(x): 
                        y = y1
                        result = coefficient_function(x, y) * boundary_function(x, y) * basis_function(x, y, vertices, k, derivative_order_x=0, derivative_order_y=0)
                        return result
                # Gaussian Integral
                integral = Gaussian_Integral_1D(lower_bound, upper_bound, f_integral, Gaussian_Integral_1D_N)
                b[Tb[j, k], 0] += integral
    return A, b


def treat_Robin_boundary_2D(P, T, Pb, Tb, A, b, boundary_edges, coefficient_function, boundary_function, basis_function, Gaussian_Integral_1D_N):
    # the number of local basis functions
    Nl = Tb.shape[1]
    # the number of finite elements
    Nb = T.shape[0]
    # the number of boundary nodes
    N_nodes = boundary_edges.shape[0]
    for i in range(N_nodes):
        if boundary_edges[i, 0] == Boundary.ROBIN_BOUNDARY:
            j = boundary_edges[i, 1]
            index1 = boundary_edges[i, 2]
            index2 = boundary_edges[i, 3]    
            x1 = Pb[index1, 0]
            y1 = Pb[index1, 1]
            x2 = Pb[index2, 0]
            y2 = Pb[index2, 1]
            vertices = np.array([[P[T[j, 0], 0], P[T[j, 0], 1]], [P[T[j, 1], 0], P[T[j, 1], 1]], [P[T[j, 2], 0], P[T[j, 2], 1]]])
            for k in range(Nl):                    
                if x1==x2:
                    if y1<=y2:
                        lower_bound = y1
                        upper_bound = y2
                    else:
                        lower_bound = y2
                        upper_bound = y1
                    def f_integral(y): 
                        x = x1
                        result = coefficient_function(x, y) * boundary_function(x, y, type=1) * basis_function(x, y, vertices, k, derivative_order_x=0, derivative_order_y=0)
                        return result
                elif y1==y2:
                    if x1<=x2:
                        lower_bound = x1
                        upper_bound = x2
                    else:
                        lower_bound = x2
                        upper_bound = x1
                    def f_integral(x): 
                        y = y1
                        result = coefficient_function(x, y) * boundary_function(x, y, type=1) * basis_function(x, y, vertices, k, derivative_order_x=0, derivative_order_y=0)
                        return result
                # Gaussian Integral
                integral = Gaussian_Integral_1D(lower_bound, upper_bound, f_integral, Gaussian_Integral_1D_N)
                b[Tb[j, k], 0] += integral
                
                for l in range(Nl):                
                    if x1==x2:
                        def f_integral(y): 
                            x = x1
                            result = coefficient_function(x, y) * boundary_function(x, y, type=0) * \
                                basis_function(x, y, vertices, k, derivative_order_x=0, derivative_order_y=0) * \
                                basis_function(x, y, vertices, l, derivative_order_x=0, derivative_order_y=0)
                            return result
                    elif y1==y2:
                        def f_integral(x): 
                            y = y1
                            result = coefficient_function(x, y) * boundary_function(x, y, type=0) * \
                                basis_function(x, y, vertices, k, derivative_order_x=0, derivative_order_y=0) * \
                                basis_function(x, y, vertices, l, derivative_order_x=0, derivative_order_y=0)
                            return result
                    integral = Gaussian_Integral_1D(lower_bound, upper_bound, f_integral, Gaussian_Integral_1D_N)
                    A[Tb[j, l], Tb[j, k]] += integral
    return A, b