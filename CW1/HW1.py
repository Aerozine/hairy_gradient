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
from getObjFVal import getObjFVal , getObjFGrad , getObjFHess
from plotOptimizationPath import plotOptimizationPath
### Parameters ###

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

# how should we set h ? 
def linear_search(x, s, functionID, h=0.1, rho=0.5, tol=Epsilon):
    # Dichotomy method ( 125)
    # rho = 0.5 -> bisection ! 
    def dphi(alpha):
        return getObjFGrad(x + alpha*s, functionID) @ s
    amin, amax = 0.0, h
    # determine the first interval (127)
    while dphi(amax) < 0:
        amin, amax = amax, 2*amax
    # each iteration (126)
    while amax - amin > tol:
        a = amax - rho*(amax- amin)
        amin, amax = (a, amax) if dphi(a) < 0 else (amin, a)
    a = amax - rho*(amax- amin)
    # phi not necessary unimodal, shrink required (140)
    # f2 is not convex
    while getObjFVal(x + a*s, functionID) >= getObjFVal(x, functionID) and a > tol:
        a *= rho
    return a
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

    H = np.eye(n) # guess from 162
    # should we put a "nice" value from default for this one ? 
    Fgrad = getObjFGrad(x[:, 0], functionID)
 
 # should be MaxIter-1 since initialization of x= n,MaxIter
    for k in range(MaxIter):
        # Bingo !
        if np.linalg.norm(Fgrad) < Epsilon:
            print("Bingo !")
            break 
        # Quasi-Newton update (151)
        s = - H @ Fgrad 
        a = linear_search(x[:, k], s, functionID) 
        # x_(K+1)= x_k + alpha * s
        x[:, k + 1] = x[:, k] + a*s

        Fgrad_new = getObjFGrad(x[:, k + 1], functionID)
        # (161)
        delta = x[:, k + 1] - x[:, k]
        gamma = Fgrad_new - Fgrad
        # dg and Hg to not compute the value a lot of times
        dg = delta @ gamma
        # assure SPD for i+1 (161)
        if dg > 0 :
            Hg = H @ gamma
            # formula slide 176 
            H = H + (1 + (gamma @ Hg)/dg) * np.outer(delta, delta)/dg - (np.outer(delta, Hg) + np.outer(Hg, delta))/dg
        Fgrad = Fgrad_new

    x = x[:, :k + 1]  # Remove the zero elements due to the initialization step

# Plot the function with the optimization path and the results

print(f'The optimal point is: x = {x[0, -1]:.5f}, y = {x[1, -1]:.5f}.')
print(f'The objective function value is: {getObjFVal(x[:, -1], functionID):.5f}.')
plotOptimizationPath(x,functionID)



