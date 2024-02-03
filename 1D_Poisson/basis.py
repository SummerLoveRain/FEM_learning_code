class Basis:
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'
    CUBIC = 'cubic'

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