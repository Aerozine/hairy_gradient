###########################################################################
#          MECA0027 Structural and Multidisciplinary Optimization         #
#                     Unconstrained Optimization                          #
#                    University Of Liège, Belgium                         #
###########################################################################

# Solve the minimization problem
#
#             min f(x,y)
#
# using the following optimization methods
#
# 1) Steepest descent
# 2) Conjugate gradients with Fletcher-Reeves update rule
# 3) BFGS Quasi-Newton


import numpy as np
from getObjFVal import getObjFVal
from plotOptimizationPath import plotOptimizationPath

### Parameters ###

functionID = 1
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

    print("You chose the steepest descent method.")

    for i in range(MaxIter):

        # ---------------------------------------------------------------------------
        # ADD YOUR CODE
        pass  # Remove this 'pass' statement once you've added your code

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 2:

    print("You chose the conjugate gradients method with Fletcher-Reeves update rule.")

    for i in range(MaxIter):

        # ---------------------------------------------------------------------------
        # ADD YOUR CODE
        pass  # Remove this 'pass' statement once you've added your code

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 3:

    print("You chose the BFGS Quasi-Newton method.")

    for i in range(MaxIter):

        # ---------------------------------------------------------------------------
        # ADD YOUR CODE
        pass  # Remove this 'pass' statement once you've added your code

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

# Plot the function with the optimization path and the results

print(f'The optimal point is: x = {x[0, -1]:.5f}, y = {x[1, -1]:.5f}.')
print(f'The objective function value is: {getObjFVal(x[:, -1], functionID):.5f}.')
plotOptimizationPath(x,functionID)



