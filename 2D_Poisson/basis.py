class Basis:
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'

def basis_reference_linear_1D(x, basis_index, derivative_order):
    '''
    x : coordinate
    basis_index : 0 left, 1 right
    derivative_order : the derivative order of some basis function
    return : the value of a basis function with some derivate order at point x which is in [0, 1]
    '''
    if basis_index == 0:
        if derivative_order == 0:
            result = 1 - x
        elif derivative_order == 1:
            result = -1
        elif derivative_order >=2 & type(derivative_order)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 1:        
        if derivative_order == 0:
            result = x
        elif derivative_order == 1:
            result = 1
        elif derivative_order >=2 & type(derivative_order)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    else:        
        ValueError("The index of basis function is not correct!")
    return result

def basis_reference_quadratic_1D(x, basis_index, derivative_order):
    '''
    x : coordinate
    basis_index : 0 left, 1 middle, 2 right
    derivative_order : the derivative order of some basis function
    return : the value of a basis function with some derivate order at point x which is in [0, 1]
    '''
    if basis_index == 0:
        if derivative_order == 0:
            result = 2*x**2 - 3*x + 1
        elif derivative_order == 1:
            result = 4*x - 3
        elif derivative_order ==2:
            result = 4
        elif derivative_order >=3 & type(derivative_order)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 1:        
        if derivative_order == 0:
            result = 2*x**2 - x
        elif derivative_order == 1:
            result = 4*x - 1
        elif derivative_order ==2:
            result = 4
        elif derivative_order >=3 & type(derivative_order)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 2:        
        if derivative_order == 0:
            result = -4*x**2 + 4*x
        elif derivative_order == 1:
            result = -8*x + 4
        elif derivative_order ==2:
            result = -8
        elif derivative_order >=3 & type(derivative_order)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    else:        
        ValueError("The index of basis function is not correct!")
    return result


# def basis_reference_cubic_1D(x, basis_index, derivative_order):
#     '''
#     x : coordinate
#     basis_index : 0 left, 1 middle, 2 right
#     derivative_order : the derivative order of some basis function
#     return : the value of a hermite basis function with some derivate order at point x which is in [0, 1]
#     '''
#     if basis_index == 0:
#         if derivative_order == 0:
#             result = 2*x**3 - 3*x**2 + 1
#         elif derivative_order == 1:
#             result = 6*x**2 - 6*x
#         elif derivative_order ==2:
#             result = 12*x - 6
#         elif derivative_order ==3:
#             result = 12
#         elif derivative_order >=4 & type(derivative_order)==int:
#             result = 0
#         else:
#             ValueError("The derivative order is not correct!")
#     elif basis_index == 1:        
#         if derivative_order == 0:
#             result = -2*x**3 + 3*x**2
#         elif derivative_order == 1:
#             result = -6*x**2 + 6*x
#         elif derivative_order ==2:
#             result = -12*x + 6
#         elif derivative_order ==3:
#             result = -12
#         elif derivative_order >=4 & type(derivative_order)==int:
#             result = 0
#         else:
#             ValueError("The derivative order is not correct!")
#     elif basis_index == 2:        
#         if derivative_order == 0:
#             result = x**3 - 2*x**2 + x
#         elif derivative_order == 1:
#             result = 3*x**2 - 4*x + 1
#         elif derivative_order ==2:
#             result = 6*x - 4
#         elif derivative_order ==3:
#             result = 6
#         elif derivative_order >=4 & type(derivative_order)==int:
#             result = 0
#         else:
#             ValueError("The derivative order is not correct!")
#     elif basis_index == 3:        
#         if derivative_order == 0:
#             result = x**3 - x**2
#         elif derivative_order == 1:
#             result = 3*x**2 - 2*x
#         elif derivative_order ==2:
#             result = 6*x - 2
#         elif derivative_order ==3:
#             result = 6
#         elif derivative_order >=4 & type(derivative_order)==int:
#             result = 0
#         else:
#             ValueError("The derivative order is not correct!")
#     else:        
#         ValueError("The index of basis function is not correct!")
#     return result


def affine_x(a, b, x, derivative_order):    
    '''
    a : the lower bound.
    b : the upper bound.
    x : coordinate
    derivative_order : the derivative order of some basis function
    return : a affine mapping value at x
    '''
    if derivative_order == 0:
        result = (x-a)/(b-a)
    elif derivative_order == 1:
        result = 1/(b-a)
    elif derivative_order >=2 & type(derivative_order)==int:
        result = 0
    else:
        ValueError("The derivative order is not correct!")
    return result

def basis_triangular_reference_linear_2D(x, y, basis_index, derivative_order_x, derivative_order_y):
    '''
    x : coordinate
    y : coordinate
    basis_index : 0 (0, 0), 1 (1, 0), 2 (0, 1)
    derivative_order : the derivative order of some basis function
    return : the value of a basis function with some derivate order at point x which is in [0, 1]
    '''
    if basis_index == 0:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = -x-y+1
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = -1
        elif derivative_order_x == 0 and derivative_order_y == 1:
            result = -1
        elif (derivative_order_x >=2 or derivative_order_y >=2) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 1:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = x
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = 1
        elif derivative_order_x == 0 and derivative_order_y == 1:
            result = 0
        elif (derivative_order_x >=2 or derivative_order_y >=2) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 2:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = y
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = 0
        elif derivative_order_x == 0 and derivative_order_y == 1:
            result = 1
        elif ((derivative_order_x + derivative_order_y) >=2) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    else:        
        ValueError("The index of basis function is not correct!")
    return result


def basis_triangular_reference_quadratic_2D(x, y, basis_index, derivative_order_x, derivative_order_y):
    '''
    x : coordinate
    y : coordinate
    basis_index : 0 (0, 0), 1 (1, 0), 2 (0, 1)
    derivative_order : the derivative order of some basis function
    return : the value of a basis function with some derivate order at point x which is in [0, 1]
    '''
    if basis_index == 0:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = 2*x**2 + 2*y**2 + 4*x*y - 3*y - 3*x + 1
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = 4*x + 4*y - 3
        elif derivative_order_x == 0 and derivative_order_y == 1:
            result = 4*y + 4*x - 3
        elif (derivative_order_x + derivative_order_y) == 2:
            result = 4
        elif ((derivative_order_x + derivative_order_y) >=3) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 1:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = 2*x**2 - x
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = 4*x
        elif derivative_order_x == 2 and derivative_order_y == 0:
            result = 4
        elif ((derivative_order_y > 0) or (derivative_order_x > 2)) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 2:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = 2*y**2 - y
        elif derivative_order_x == 0 and derivative_order_y == 1:
            result = 4*y - 1
        elif derivative_order_x == 0 and derivative_order_y == 2:
            result = 4
        elif ((derivative_order_y > 2) or (derivative_order_x > 0)) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 3:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = -4*x**2 - 4*x*y + 4*x
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = -8*x - 4*y + 4
        elif derivative_order_x == 0 and derivative_order_y == 1:
            result = -4*x
        elif derivative_order_x == 1 and derivative_order_y == 1:
            result = -4
        elif derivative_order_x == 2 and derivative_order_y == 0:
            result = -8
        elif derivative_order_x == 0 and derivative_order_y == 2:
            result = 0
        elif ((derivative_order_x + derivative_order_y >= 3)) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 4:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = 4*x*y
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = 4*y
        elif derivative_order_x == 0 and derivative_order_y == 1:
            result = 4*x
        elif derivative_order_x == 1 and derivative_order_y == 1:
            result = 4
        elif derivative_order_x == 2 and derivative_order_y == 0:
            result = 0
        elif derivative_order_x == 0 and derivative_order_y == 2:
            result = 0
        elif ((derivative_order_x + derivative_order_y >= 3)) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 5:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = -4*y**2 - 4*x*y + 4*y
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = -4*y
        elif derivative_order_x == 0 and derivative_order_y == 1:
            result = -8*y - 4*x + 4
        elif derivative_order_x == 1 and derivative_order_y == 1:
            result = -4
        elif derivative_order_x == 2 and derivative_order_y == 0:
            result = 0
        elif derivative_order_x == 0 and derivative_order_y == 2:
            result = -8
        elif ((derivative_order_x + derivative_order_y >= 3)) & type(derivative_order_x)==int & type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    else:        
        ValueError("The index of basis function is not correct!")
    return result


def affine_triangular_xy(x, y, vertices, derivative_order_x, derivative_order_y):    
    '''
    a : the lower bound.
    b : the upper bound.
    x : coordinate
    y : coordinate
    vertices: the coordinates of the triangle
    derivative_order_x : the derivative order of some basis function about x
    derivative_order_y : the derivative order of some basis function about y
    return : a affine mapping value at (x, y)
    '''
    x1 = vertices[0, 0]
    y1 = vertices[0, 1]
    x2 = vertices[1, 0]
    y2 = vertices[1, 1]
    x3 = vertices[2, 0]
    y3 = vertices[2, 1]
    Jx1 = x2-x1
    Jx2 = x3-x1
    Jy1 = y2-y1
    Jy2 = y3-y1
    J = Jx1*Jy2 - Jx2*Jy1
    result = []
    if derivative_order_x == 0 and derivative_order_y == 0:
        x_hat = (Jy2*(x-x1)-Jx2*(y-y1))/J
        y_hat = (-Jy1*(x-x1)+Jx1*(y-y1))/J
        result.append(x_hat)
        result.append(y_hat)
    elif derivative_order_x == 1 and derivative_order_y == 0:
        x_hat = Jy2/J
        y_hat = -Jy1/J
        result.append(x_hat)
        result.append(y_hat)
    elif derivative_order_x == 0 and derivative_order_y == 1:
        x_hat = -Jx2/J
        y_hat = Jx1/J
        result.append(x_hat)
        result.append(y_hat)
    elif derivative_order_x == 2 and derivative_order_y == 0:
        x_hat = Jy2**2/J**2
        y_hat = 2*(-Jy2*Jy1)/J**2
        z_hat = Jy1**2/J**2
        result.append(x_hat)
        result.append(y_hat)
        result.append(z_hat)
    elif derivative_order_x == 1 and derivative_order_y == 1:
        x_hat = Jx2**2/J**2
        y_hat = 2*(-Jx2*Jx1)/J**2
        z_hat = Jx1**2/J**2
        result.append(x_hat)
        result.append(y_hat)
        result.append(z_hat)
    elif derivative_order_x == 0 and derivative_order_y == 2:
        x_hat = -Jx2*Jy2/J**2
        y_hat = (Jy2*Jy1 + Jx1*Jy2)/J**2
        z_hat = -Jx1*Jy1/J**2
        result.append(x_hat)
        result.append(y_hat)
        result.append(z_hat)
    elif (derivative_order_x + derivative_order_y) >=3 & type(derivative_order_x)==int & type(derivative_order_y)==int:
        result = 0
    else:
        ValueError("The derivative order is not correct!")
    return result