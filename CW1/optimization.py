import numpy as np
from getObjFVal import *
from linear import linear_search

#basically create an error called MaxIter so the code stop and we can identify it 
class MaxIterError(RuntimeError):
    pass


#p61
def crit_1_max(f_k, f_kp1, grad_k, grad_kp1):
    
    crit = np.max(np.abs(grad_kp1)) 
    
    return crit

def crit_2_norm(f_k, f_kp1, grad_k, grad_kp1):
    # returns the square root not the 
    #crit = np.linalg.norm(grad_kp1) 
    crit = np.dot(grad_kp1,grad_kp1)
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
        if criterion == crit_3_diff:
            # |f(x^k) - f(x^k-1)| needs two iterates, so it cannot be evaluated
            # at k = 0: x[:, -1] is the last (still empty) column, not x^-1.
            # f_k is kept from the previous pass instead of recomputed.
            f_kp1 = getObjFVal(x[:, k], functionID)
            if k > 0 and criterion(f_k, f_kp1, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                break
            f_k = f_kp1
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                break
        alpha[k] = linear_search(x[:, k], grad[:, k], functionID, Epsilon)

        x[:, k + 1] = x[:, k] + alpha[k] * grad[:, k]

    # If the for loop finished, MaxIter is exceeeded (otherwise break before)
    else:
        raise MaxIterError(
            f"steepest descent: MaxIter = {MaxIter} reached without satisfying "
            f"{criterion.__name__} < {Epsilon:g}, last ||grad|| = "
            f"{np.linalg.norm(grad[:, k]):.3e}"
        )

    return x[:, : k + 1]  # Remove the zero elements due to the initialization step


# 2) Conjugate gradients with Fletcher-Reeves update rule
# restart = 0 -> no re-initialization, restart = n -> the course's rule (110)
def conjugate_gradients_Fletcher_Reeves(
    x, n, functionID, MaxIter, Epsilon,criterion = crit_2_norm, restart=3, verbose=True
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
        
        k = i
        if criterion == crit_3_diff:
            # |f(x^k) - f(x^k-1)| needs two iterates, so it cannot be evaluated
            # at k = 0: x[:, -1] is the last (still empty) column, not x^-1.
            # f_k is kept from the previous pass instead of recomputed.
            f_kp1 = getObjFVal(x[:, k], functionID)
            if k > 0 and criterion(f_k, f_kp1, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                break
            f_k = f_kp1
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                break

        # FUNDAMENTAL REMARK (110)
        # the global convergence is submitted to a periodic re-initialization
        # -> after n iterations, go on from x^n with the direction -grad f(x^n)
        if restart and i > 0 and i % restart == 0:
            s[:, i] = -grad_x

        # f2 is general C1 -> need for a Line Search method
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
        raise MaxIterError(
            f"conjugate gradients: MaxIter = {MaxIter} reached without satisfying "
            f"{criterion.__name__} < {Epsilon:g}, last ||grad|| = "
            f"{np.linalg.norm(grad[:, i]):.3e}"
        )
    return x[:, : i + 1]  # Remove the zero elements due to the initialization step


# 3) BFGS Quasi-Newton
# restart = 0 -> no re-initialization, restart = n -> the course's rule (162)
def BFGS(x, n, functionID, MaxIter, Epsilon,criterion = crit_2_norm, restart=0, verbose=True):
    if verbose:
        print("You chose the BFGS Quasi-Newton method.")
        
    grad = np.zeros((n, MaxIter))

    H = np.eye(n)  # guess from 162
    # should we put a "nice" value from default for this one ?
    Fgrad = getObjFGrad(x[:, 0], functionID)

    # should be MaxIter-1 since initialization of x= n,MaxIter
    for k in range(MaxIter - 1):
        grad[:,k] = Fgrad
        
        if criterion == crit_3_diff:
            # |f(x^k) - f(x^k-1)| needs two iterates, so it cannot be evaluated
            # at k = 0: x[:, -1] is the last (still empty) column, not x^-1.
            # f_k is kept from the previous pass instead of recomputed.
            f_kp1 = getObjFVal(x[:, k], functionID)
            if k > 0 and criterion(f_k, f_kp1, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                break
            f_k = f_kp1
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k]) < Epsilon:
                if verbose:
                    print("Bingo !")
                break
        
        
        # periodic re-initialization (162)
        # -> after n iterations, starting from x^n, go on with H = I
        if restart and k > 0 and k % restart == 0:
            H = np.eye(n)

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

    # If the for loop finished, MaxIter is exceeeded (otherwise break before)
    else:
        raise MaxIterError(
            f"BFGS: MaxIter = {MaxIter} reached without satisfying "
            f"{criterion.__name__} < {Epsilon:g}, last ||grad|| = "
            f"{np.linalg.norm(Fgrad):.3e}"
        )

    return x[:, : k + 1]  # Remove the zero elements due to the initialization step

# mapping from name to function name basically
methods = {
    "Steepest descent": steepest_descent,
    "Conjugate gradients": conjugate_gradients_Fletcher_Reeves,
    "BFGS": BFGS,
}
