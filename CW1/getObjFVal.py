import numpy as np
def getObjFVal(x, functionID):
    x, y = x[0], x[1]
    if functionID == 1:
        fval = 5*(x +1 )**2 - 4 *x*y + 3*y**2 - 2 *x + 7*y +2
    elif functionID == 2:
        fval = 0.2*x**2 +0.3 * y**2 - 0.2 *x*y - 15 * np.cos(0.7*x)-9*np.sin(0.6*y-0.7*x)+ 3*x 
    else:
        raise ValueError("Unknown functionID")

    return fval

def getObjFGrad(x,functionID):
    x, y = x[0], x[1]
    if functionID == 1:
        return np.array([10*x - 4*y + 8,
                        -4*x + 6*y + 7])
    elif functionID ==2:
        return np.array([0.4*x - 0.2*y + 10.5*np.sin(0.7*x) + 6.3*np.cos(0.7*x - 0.6*y) + 3,
                     -0.2*x + 0.6*y - 5.4*np.cos(0.7*x - 0.6*y)])
    raise ValueError("Unknown functionID")

def getObjFHess(x,functionID):
    x, y = x[0], x[1]
    if functionID ==1 :
        return np.array([[10.0, -4.0],
                        [-4.0, 6.0]]) 
    elif functionID ==2 :
        return np.array([[-4.41*np.sin(0.7*x - 0.6*y) + 7.35*np.cos(0.7*x) + 0.4, 3.78*np.sin(0.7*x - 0.6*y) - 0.2],
                     [3.78*np.sin(0.7*x - 0.6*y) - 0.2, 0.6 - 3.24*np.sin(0.7*x - 0.6*y)]])
    raise ValueError("Unknown functionID")
