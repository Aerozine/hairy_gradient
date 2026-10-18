import numpy as np
from getObjFVal import *
from linear import linear_search

#p61
def crit_1_max(f_k, f_kp1, grad_k, grad_kp1):
    
    crit = np.max(np.abs(grad_kp1)) 
    
    return crit

def crit_2_norm(f_k, f_kp1, grad_k, grad_kp1):
    
    crit = np.dot(grad_kp1, grad_kp1) #mettre k+1 ???
    
    return crit

def crit_3_diff(f_k, f_kp1, grad_k, grad_kp1):
    
    crit = np.abs(f_kp1-f_k)

    return crit



def steepest_descent(x, n, functionID, MaxIter, Epsilon,criterion = crit_2_norm, verbose=True):
    if verbose:
        print("You chose the steepest descent method.")

    grad = np.zeros((n, MaxIter))
    alpha = np.zeros(MaxIter)

    for k in range(MaxIter - 1):  # k+1 has to stay a column of x
        grad[:, k] = -getObjFGrad(x[: ,k], functionID)
        
        # grad[:,i] already holds MINUS the gradient, i.e. the descent direction
        # Bingo !
        '''
        if criterion(_,_, grad[:,k-1], grad[:,k]) < Epsilon:
            if verbose:
                print("Bingo !")
                break
        # black magic artifact allowing us to dont pollute 
        # the number of call of the objective function inside the call
        '''
        
        if criterion == crit_3_diff:    
            f_k = getObjFVal(x[:, k - 1], functionID)   #maitre loic est content ?
            f_kp1 = getObjFVal(x[:, k], functionID)
            if criterion(f_k, f_kp1, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                    break
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                    break
        
        
        
        alpha[k] = linear_search(x[:, k], grad[:, k], functionID, Epsilon)

        x[:, k + 1] = x[:, k] + alpha[k] * grad[:, k]
        
    # If the for loop finished, MaxIter is exceeeded (otherwise break before)
    if k == MaxIter :
        if verbose:
            print("MaxIter exceeded")

    return x[:, : k + 1]  # Remove the zero elements due to the initialization step


# 2) Conjugate gradients with Fletcher-Reeves update rule
def conjugate_gradients_Fletcher_Reeves(
    x, n, functionID, MaxIter, Epsilon,criterion = crit_2_norm, verbose=True
):
    if verbose:
        print(
            "You chose the conjugate gradients method with Fletcher-Reeves update rule."
        )

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
    grad = np.zeros((n, MaxIter))

    for i in range(MaxIter - 1):

        # Compute current gradient
        grad_x = Fgrad(x[:, i])
        grad[:,i] = grad_x 
        
        '''
        # Bingo!
        if np.linalg.norm(grad_x) < Epsilon:
            if verbose:
                print("Bingo !")
            break
        '''
        k = i
        if criterion == crit_3_diff:    
            f_k = getObjFVal(x[:, k - 1], functionID)   #maitre loic est content ?
            f_kp1 = getObjFVal(x[:, k], functionID)
            if criterion(f_k, f_kp1, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                    break
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                    break

        # f2 is general C1 -> need for a Line Search method
        """Should we implement quadratic function method ?"""
        """Discussion with other LS methods ? """
        alpha[i] = linear_search(x[:, i], s[:, i], functionID, Epsilon)

        # Update point x (108)
        x[:, i + 1] = x[:, i] + alpha[i] * s[:, i]

        # Compute next iterate gradient
        next_grad_x = Fgrad(x[:, i + 1])

        # Fletcher and Reeves update rule (109)
        beta[i] = (np.linalg.norm(next_grad_x) ** 2) / (np.linalg.norm(grad_x) ** 2)

        # Update search direction (109)
        s[:, i + 1] = -next_grad_x + beta[i] * s[:, i]

    # If the for loop finished, MaxIter is exceeeded (otherwise break before)
    else:
        if verbose:
            print("MaxIter exceeded")
    return x[:, : i + 1]  # Remove the zero elements due to the initialization step


# 3) BFGS Quasi-Newton
def BFGS(x, n, functionID, MaxIter, Epsilon,criterion = crit_2_norm, verbose=True):
    if verbose:
        print("You chose the BFGS Quasi-Newton method.")
        
    grad = np.zeros((n, MaxIter))

    H = np.eye(n)  # guess from 162
    # should we put a "nice" value from default for this one ?
    Fgrad = getObjFGrad(x[:, 0], functionID)

    # should be MaxIter-1 since initialization of x= n,MaxIter
    for k in range(MaxIter - 1):
        # Bingo !
        '''
        if np.linalg.norm(Fgrad) < Epsilon:
            if verbose:
                print("Bingo !")
            break
        '''
        grad[:,k] = Fgrad
        
        if criterion == crit_3_diff:    
            f_k = getObjFVal(x[:, k - 1], functionID)   #maitre loic est content ?
            f_kp1 = getObjFVal(x[:, k], functionID)
            if criterion(f_k, f_kp1, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                    break
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                    break
        
        
        # Quasi-Newton update (151)
        s = -H @ Fgrad
        a = linear_search(x[:, k], s, functionID, Epsilon)
        # x_(K+1)= x_k + alpha * s
        x[:, k + 1] = x[:, k] + a * s

        Fgrad_new = getObjFGrad(x[:, k + 1], functionID)
        # (161)
        delta = x[:, k + 1] - x[:, k]
        gamma = Fgrad_new - Fgrad
        # dg and Hg to not compute the value a lot of times
        dg = delta @ gamma
        # assure SPD for i+1 (161)
        if dg > 0:
            Hg = H @ gamma
            # formula slide 176
            H = (
                H
                + (1 + (gamma @ Hg) / dg) * np.outer(delta, delta) / dg
                - (np.outer(delta, Hg) + np.outer(Hg, delta)) / dg
            )
        Fgrad = Fgrad_new
    return x[:, : k + 1]  # Remove the zero elements due to the initialization step
