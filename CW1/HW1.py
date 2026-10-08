###########################################################################
#          MECA0027 Structural and Multidisciplinary Optimization         #
#                     Unconstrained Optimization                          #
#                    University Of Liège, Belgium                         #
###########################################################################

import numpy as np
from linear import *
from optimization import *
from getObjFVal import getObjFVal, getObjFGrad, getObjFHess
from plotOptimizationPath import plotOptimizationPath

functionID = 2
xinit = np.array([0, 0])  # initial point
MaxIter = 100  # Maximum number of iterations
Epsilon = 1e-5  # Tolerance for the stop criteria

### Initialization ###

print("Which optimization method do you want to use? Press:")
print("     1 for Steepest descent method")
print("     2 for Conjugate gradients method with Fletcher-Reeves update rule")
print("     3 for BFGS Quasi-Newton method")
method = int(input())

n = 2  # Dimension of the problem
xinit = xinit.reshape(2, 1)  # To be sure that it's a column vector
x = np.zeros((n, MaxIter))  # Initialization of vector x
x[:, 0] = xinit[:, 0]  # Put xinit in vector x.


### Methods ###

if method == 1:
    x = steepest_descent(x, n, functionID, MaxIter, Epsilon)
elif method == 2:
    x = conjugate_gradients_Fletcher_Reeves(x, n, functionID, MaxIter, Epsilon)
elif method == 3:
    x = BFGS(x, n, functionID, MaxIter, Epsilon)

# Plot the function with the optimization path and the results

print(f"The optimal point is: x = {x[0, -1]:.5f}, y = {x[1, -1]:.5f}.")
print(f"The objective function value is: {getObjFVal(x[:, -1], functionID):.5f}.")
plotOptimizationPath(x, functionID)
