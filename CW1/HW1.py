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

# Line search method 2: Newton-Raphson (117)

# x, s and alpha are current iterates i in the method calling the function
def newton_raphson(x, s, alpha, functionID, tol=Epsilon):
    
    def phi(alpha):
        return getObjFVal(x + alpha * s, functionID)
    
    def dphi(alpha):
        return getObjFGrad(x + alpha * s, functionID) @ s
    
    def ddphi(alpha):
        return s.T @ getObjFHess(x + alpha * s, functionID) @ s
    
    while abs(dphi(alpha)) > tol:
        alpha -= dphi(alpha)/ddphi(alpha)

    return alpha

### Methods ###

if method == 1:

    print("You chose the steepest descent method.")

    grad = np.zeros((n, MaxIter))
    alpha = np.zeros(MaxIter)
    
    for i in range(MaxIter):
        
        #Step 1 - Already initialize
        #i = k
        
        #Step 2
        if functionID == 1 : 
            grad[0,i] = -(10*x[0,i] - 4*x[1,i] + 8)
            grad[1,i] = -(-4*x[0,i] + 6*x[1,i] + 7)
            
        if functionID == 2 : 
            grad[0,i] = -(0.4*x[0,i] - 0.2*x[1,i] + 10.5*np.sin(0.7*x[0,i]) + 6.3*np.cos(0.7*x[0,i] - 0.6*x[1,i]) + 3)                                                       
            grad[1,i] = -(-0.2*x[0,i] + 0.6*x[1,i] - 5.4*np.cos(0.7*x[0,i] - 0.6*x[1,i]))
        
        #Step 3
        # alpha_it = np.linspace(0, 1000, 1e6)
        # fct = getObjFVal(x[:, i] + alpha_it*grad[:,i], functionID)
        # min_fct_alpha = np.min(fct)
        # alpha[i] = alpha_it[fct.index(min_fct_alpha)]
        
        alpha[i] = linear_search(x, s, functionID, h=0.1, rho=0.5, Epsilon)
        
        #Step 4
        x[:, i+1] = x[:, i] + alpha[i]*grad[:,i]
        
        if (alpha[i]*grad[:,i] < Epsilon) : 
            break

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 2:

    print("You chose the conjugate gradients method with Fletcher-Reeves update rule.")
    
    # Shortcut function
    def Fgrad(x):
        return getObjFGrad(x, functionID)

    # Initialize parameters alpha and beta
    alpha = np.zeros(MaxIter)
    beta = np.zeros(MaxIter)

    # Initialize search direction
    sinit = -getObjFGrad(x[:, 0], functionID)  
    s = np.zeros((n, MaxIter))
    s[:, 0] = sinit

    for i in range(MaxIter - 1):
        
        # Bingo!
        if np.linalg.norm(Fgrad(x[:, i])) < Epsilon:
            print("Bingo !")
            break 
            
        # f2 is general C1 -> need for a Line Search method
        """Should we implement quadratic function method ?"""
        """Discussion with other LS methods ? """
        alpha[i] = linear_search(x[:,i], s[:,i], functionID)
        
        # Update point x (108)
        x[:, i + 1] = x[:, i] + alpha[i] * s[:,i]
        
        # Fletcher and Reeves update rule (109)
        beta[i] = (np.linalg.norm(Fgrad(x[:, i + 1])) ** 2)/(np.linalg.norm(Fgrad(x[:, i])) ** 2)
        
        # Update search direction (109)
        s[:, i + 1] = -Fgrad(x[:, i + 1]) + beta[i] * s[:, i]
    
    # If the for loop finished, MaxIter is exceeeded (otherwise break before)
    else:
        print("MaxIter exceeded")

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 3:
    print("You chose the BFGS Quasi-Newton method.")

    H = np.eye(n) # guess from 162
    # should we put a "nice" value from default for this one ? 
    Fgrad = getObjFGrad(x[:, 0], functionID)
 
 # should be MaxIter-1 since initialization of x= n,MaxIter
    for k in range(MaxIter):
        # Bingo !
        if np.linalg.norm(Fgrad) < Epsilon: # Fgrad est calculé en x_0, non? 
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
