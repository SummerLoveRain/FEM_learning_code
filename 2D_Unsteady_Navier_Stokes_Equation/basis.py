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

def reference_linear_triangular_2D_basis(x, y, basis_index, derivative_order_x, derivative_order_y):
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
        elif (derivative_order_x + derivative_order_y >=2) and type(derivative_order_x)==int and type(derivative_order_y)==int:
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
        elif ((derivative_order_x + derivative_order_y) >=2) and type(derivative_order_x)==int and type(derivative_order_y)==int:
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
        elif ((derivative_order_x + derivative_order_y) >=2) and type(derivative_order_x)==int and type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    else:        
        ValueError("The index of basis function is not correct!")
    return result


def reference_quadratic_triangular_2D_basis(x, y, basis_index, derivative_order_x, derivative_order_y):
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
        elif ((derivative_order_x + derivative_order_y) >=3) and type(derivative_order_x)==int and type(derivative_order_y)==int:
            result = 0
        else:
            ValueError("The derivative order is not correct!")
    elif basis_index == 1:
        if derivative_order_x == 0 and derivative_order_y == 0:
            result = 2*x**2 - x
        elif derivative_order_x == 1 and derivative_order_y == 0:
            result = 4*x - 1
        elif derivative_order_x == 2 and derivative_order_y == 0:
            result = 4
        elif ((derivative_order_y > 0) or (derivative_order_x > 2)) and type(derivative_order_x)==int and type(derivative_order_y)==int:
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
        elif ((derivative_order_y > 2) or (derivative_order_x > 0)) and type(derivative_order_x)==int and type(derivative_order_y)==int:
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
        elif ((derivative_order_x + derivative_order_y >= 3)) and type(derivative_order_x)==int and type(derivative_order_y)==int:
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
        elif ((derivative_order_x + derivative_order_y >= 3)) and type(derivative_order_x)==int and type(derivative_order_y)==int:
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
        elif ((derivative_order_x + derivative_order_y >= 3)) and type(derivative_order_x)==int and type(derivative_order_y)==int:
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
        xhat = (Jy2*(x-x1)-Jx2*(y-y1))/J
        yhat = (-Jy1*(x-x1)+Jx1*(y-y1))/J
        result.append(xhat)
        result.append(yhat)
    elif derivative_order_x == 1 and derivative_order_y == 0:
        xhat_x = Jy2/J
        yhat_x = -Jy1/J
        result.append(xhat_x)
        result.append(yhat_x)
    elif derivative_order_x == 0 and derivative_order_y == 1:
        xhat_y = -Jx2/J
        yhat_y = Jx1/J
        result.append(xhat_y)
        result.append(yhat_y)
    elif derivative_order_x == 2 and derivative_order_y == 0:
        xhat2_x2 = Jy2**2/J**2
        xyhat_x2 = 2*(-Jy2*Jy1)/J**2
        yhat2_x2 = Jy1**2/J**2
        result.append(xhat2_x2)
        result.append(xyhat_x2)
        result.append(yhat2_x2)
    elif derivative_order_x == 1 and derivative_order_y == 1:
        xhat2_xy = -Jx2*Jy2/J**2
        xyhat_xy = (Jy2*Jy1 + Jx1*Jy2)/J**2
        yhat2_xy = -Jx1*Jy1/J**2
        result.append(xhat2_xy)
        result.append(xyhat_xy)
        result.append(yhat2_xy)
    elif derivative_order_x == 0 and derivative_order_y == 2:
        xhat2_y2 = Jx2**2/J**2
        xyhat_y2 = 2*(-Jx2*Jx1)/J**2
        yhat2_y2 = Jx1**2/J**2
        result.append(xhat2_y2)
        result.append(xyhat_y2)
        result.append(yhat2_y2)
    else:
        ValueError("The derivative order is not correct!")
    return result

def local_linear_triangular_2D_basis(x, y, vertices, basis_index, derivative_order_x, derivative_order_y):
    '''
    x : coordinate
    y : coordinate
    basis_index : 0 (0, 0), 1 (1, 0), 2 (0, 1)
    derivative_order : the derivative order of some basis function
    return : the value of a basis function with some derivate order at point x which is in [0, 1]
    '''
    new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
    if derivative_order_x == 0 and derivative_order_y == 0:
        result = reference_linear_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x, derivative_order_y)
    elif derivative_order_x == 1 and derivative_order_y == 0:
        psi_x = reference_linear_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=1, derivative_order_y=0)
        psi_y = reference_linear_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=0, derivative_order_y=1)
        xhat_x, yhat_x = affine_triangular_xy(x, y, vertices, derivative_order_x=1, derivative_order_y=0)
        result = psi_x*xhat_x + psi_y*yhat_x
    elif derivative_order_x == 0 and derivative_order_y == 1:
        psi_x = reference_linear_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=1, derivative_order_y=0)
        psi_y = reference_linear_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=0, derivative_order_y=1)
        xhat_y, yhat_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=1)
        result = psi_x*xhat_y + psi_y*yhat_y
    elif (derivative_order_x + derivative_order_y >=2) and type(derivative_order_x)==int and type(derivative_order_y)==int:
        result = 0
    else:
        ValueError("The derivative order is not correct!")
    return result


def local_quadratic_triangular_2D_basis(x, y, vertices, basis_index, derivative_order_x, derivative_order_y):
    '''
    x : coordinate
    y : coordinate
    basis_index : 0 (0, 0), 1 (1, 0), 2 (0, 1)
    derivative_order : the derivative order of some basis function
    return : the value of a basis function with some derivate order at point x which is in [0, 1]
    '''
    new_x, new_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=0)
    if derivative_order_x == 0 and derivative_order_y == 0:
        result = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x, derivative_order_y)
    elif derivative_order_x == 1 and derivative_order_y == 0:
        psi_x = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=1, derivative_order_y=0)
        psi_y = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=0, derivative_order_y=1)
        xhat_x, yhat_x = affine_triangular_xy(x, y, vertices, derivative_order_x=1, derivative_order_y=0)
        result = psi_x*xhat_x + psi_y*yhat_x
    elif derivative_order_x == 0 and derivative_order_y == 1:
        psi_x = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=1, derivative_order_y=0)
        psi_y = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=0, derivative_order_y=1)
        xhat_y, yhat_y = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=1)
        result = psi_x*xhat_y + psi_y*yhat_y
    elif derivative_order_x == 2 and derivative_order_y == 0:
        psi_xx = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=2, derivative_order_y=0)
        psi_xy = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=1, derivative_order_y=1)
        psi_yy = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=0, derivative_order_y=2)
        xhat2_x2, xyhat_x2, yhat2_x2 = affine_triangular_xy(x, y, vertices, derivative_order_x=2, derivative_order_y=0)
        result = psi_xx*xhat2_x2 + psi_xy*xyhat_x2 + psi_yy*yhat2_x2
    elif derivative_order_x == 0 and derivative_order_y == 2:
        psi_xx = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=2, derivative_order_y=0)
        psi_xy = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=1, derivative_order_y=1)
        psi_yy = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=0, derivative_order_y=2)
        xhat2_y2, xyhat_y2, yhat2_y2 = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=2)
        result = psi_xx*xhat2_y2 + psi_xy*xyhat_y2 + psi_yy*yhat2_y2
    elif derivative_order_x == 1 and derivative_order_y == 1:
        psi_xx = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=2, derivative_order_y=0)
        psi_xy = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=1, derivative_order_y=1)
        psi_yy = reference_quadratic_triangular_2D_basis(new_x, new_y, basis_index, derivative_order_x=0, derivative_order_y=2)
        xhat2_xy, xyhat_xy, yhat2_xy = affine_triangular_xy(x, y, vertices, derivative_order_x=0, derivative_order_y=2)
        result = psi_xx*xhat2_xy + psi_xy*xyhat_xy + psi_yy*yhat2_xy
    elif (derivative_order_x + derivative_order_y >=3) and type(derivative_order_x)==int and type(derivative_order_y)==int:
        result = 0
    else:
        ValueError("The derivative order is not correct!")
    return result