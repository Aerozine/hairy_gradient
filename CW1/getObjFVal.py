import numpy as np
def getObjFVal(x, functionID):
    if functionID == 1:
        fval = 5 (x +1 )**2 - 4 *x*y + 3*y**2 - 2 *x + 7*y +2
    elif functionID == 2:
        fval = 0.2*x**2 +0.3 * y**2 - 0.2 *x*y - 15 * np.cos(0.7*x)-9*np.sin(0.6*y-0.7*x)+ 3*x 
    else:
        raise ValueError("Unknown functionID")

    return fval