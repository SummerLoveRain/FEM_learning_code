class Basis:
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'
    CUBIC = 'cubic'

def reference_linear_1D_basis(x, basis_index, derivative_order):
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
        elif derivative_order >=2 and type(derivative_order)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 1:        
        if derivative_order == 0:
            result = x
        elif derivative_order == 1:
            result = 1
        elif derivative_order >=2 and type(derivative_order)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    else:        
        ValueError("The index of basis function is not correct!")
    return result

def reference_quadratic_1D_basis(x, basis_index, derivative_order):
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
        elif derivative_order >=3 and type(derivative_order)==int:
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
        elif derivative_order >=3 and type(derivative_order)==int:
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
        elif derivative_order >=3 and type(derivative_order)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    else:        
        ValueError("The index of basis function is not correct!")
    return result

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
    elif derivative_order >=2 and type(derivative_order)==int:
        result = 0
    else:
        ValueError("The derivative order is not correct!")
    return result



def local_linear_1D_basis(a, b, x, basis_index, derivative_order):
    x_hat = affine_x(a, b, x, derivative_order=0)
    if derivative_order == 0:
        result = reference_linear_1D_basis(x_hat, basis_index, derivative_order=0)
    elif derivative_order == 1:
        result = reference_linear_1D_basis(x_hat, basis_index, derivative_order=1)*affine_x(a, b, x, derivative_order=1)
    elif derivative_order >=2 and type(derivative_order)==int:
        result = 0
    return result

def local_quadratic_1D_basis(a, b, x, basis_index, derivative_order):
    x_hat = affine_x(a, b, x, derivative_order=0)
    if derivative_order == 0:
        result = reference_quadratic_1D_basis(x_hat, basis_index, derivative_order=0)
    elif derivative_order == 1:
        result = reference_quadratic_1D_basis(x_hat, basis_index, derivative_order=1)*affine_x(a, b, x, derivative_order=1)
    elif derivative_order == 2:
        result = reference_quadratic_1D_basis(x_hat, basis_index, derivative_order=2)*affine_x(a, b, x, derivative_order=1)**2
    elif derivative_order >=3 and type(derivative_order)==int:
        result = 0
    return result