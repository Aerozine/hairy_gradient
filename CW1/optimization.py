import numpy as np
from getObjFVal import *
from linear import linear_search

# 1) Steepest descent method
def steepest_descent(x, n, functionID, MaxIter, Epsilon, verbose=True):
    if verbose:
        print("You chose the steepest descent method.")

    grad = np.zeros((n, MaxIter))
    alpha = np.zeros(MaxIter)

    for i in range(MaxIter - 1):  # i+1 has to stay a column of x
        grad[:, i] = -getObjFGrad(x[: ,i], functionID)
        
        # grad[:,i] already holds MINUS the gradient, i.e. the descent direction
        # Bingo !
        if np.linalg.norm(grad[:, i]) < Epsilon:
            if verbose:
                print("Bingo !")
            break

        alpha[i] = linear_search(x[:, i], grad[:, i], functionID, Epsilon)

        x[:, i + 1] = x[:, i] + alpha[i] * grad[:, i]
        
    # If the for loop finished, MaxIter is exceeeded (otherwise break before)
    if i == MaxIter :
        if verbose:
            print("MaxIter exceeded")

    return x[:, : i + 1]  # Remove the zero elements due to the initialization step


# 2) Conjugate gradients with Fletcher-Reeves update rule
def conjugate_gradients_Fletcher_Reeves(
    x, n, functionID, MaxIter, Epsilon, verbose=True
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

    for i in range(MaxIter - 1):

        # Compute current gradient
        grad_x = Fgrad(x[:, i])

        # Bingo!
        if np.linalg.norm(grad_x) < Epsilon:
            if verbose:
                print("Bingo !")
            break

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
        if verbose:
            print("MaxIter exceeded")
    return x[:, : i + 1]  # Remove the zero elements due to the initialization step


# 3) BFGS Quasi-Newton
def BFGS(x, n, functionID, MaxIter, Epsilon, verbose=True):
    if verbose:
        print("You chose the BFGS Quasi-Newton method.")

    H = np.eye(n)  # guess from 162
    # should we put a "nice" value from default for this one ?
    Fgrad = getObjFGrad(x[:, 0], functionID)

    # should be MaxIter-1 since initialization of x= n,MaxIter
    for k in range(MaxIter - 1):
        # Bingo !
        if np.linalg.norm(Fgrad) < Epsilon:
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
