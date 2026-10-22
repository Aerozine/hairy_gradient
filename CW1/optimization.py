import sys

import numpy as np
from getObjFVal import *
from linear import linear_search

#basically create an error called MaxIter so the code stop and we can identify it 
class MaxIterError(RuntimeError):
    pass


#p61
def crit_1_max(f_k, f_kp1, grad_k, grad_kp1,Epsilon):
    
    crit = np.max(np.abs(grad_kp1)) 
    
    return crit < Epsilon

def crit_2_norm(f_k, f_kp1, grad_k, grad_kp1,Epsilon):
    # returns the square root not the 
    #crit = np.linalg.norm(grad_kp1) 
    crit = np.dot(grad_kp1,grad_kp1)
    return crit < Epsilon

def crit_3_diff(f_k, f_kp1, grad_k, grad_kp1,Epsilon):
    
    crit = np.abs(f_kp1-f_k)

    return crit < Epsilon

def crit_1_3(f_k, f_kp1, grad_k, grad_kp1,Epsilon):
    return crit_3_diff(f_k, f_kp1, grad_k, grad_kp1,Epsilon) and crit_1_max(f_k, f_kp1, grad_k, grad_kp1,Epsilon)
def crit_2_3(f_k, f_kp1, grad_k, grad_kp1,Epsilon):
    return crit_2_norm(f_k, f_kp1, grad_k, grad_kp1,Epsilon) and crit_3_diff(f_k, f_kp1, grad_k, grad_kp1,Epsilon)


# the criteria built on |f(x^k) - f(x^k-1)| need the objective value of the
# previous iterate; the purely gradient-based ones are given 0, 0 instead
NEEDS_FVAL = (crit_3_diff, crit_1_3, crit_2_3)

def steepest_descent(x, n, functionID, MaxIter, Epsilon,criterion = crit_2_norm, verbose=True):
    if verbose:
        print("You chose the steepest descent method.")

    grad = np.zeros((n, MaxIter))
    alpha = np.zeros(MaxIter)

    for k in range(MaxIter - 1):  # k+1 has to stay a column of x
        grad[:, k] = -getObjFGrad(x[: ,k], functionID)
        # grad[:,i] already holds MINUS the gradient, i.e. the descent direction
        if criterion in NEEDS_FVAL:
            # |f(x^k) - f(x^k-1)| needs two iterates, so it cannot be evaluated
            # at k = 0: x[:, -1] is the last (still empty) column, not x^-1.
            # f_k is kept from the previous pass instead of recomputed.
            f_kp1 = getObjFVal(x[:, k], functionID)
            if k > 0 and criterion(f_k, f_kp1, grad[:,k-1], grad[:,k],Epsilon):
                if verbose:
                    print("Bingo !")
                break
            f_k = f_kp1
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k], Epsilon):
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
        if criterion in NEEDS_FVAL:
            # |f(x^k) - f(x^k-1)| needs two iterates, so it cannot be evaluated
            # at k = 0: x[:, -1] is the last (still empty) column, not x^-1.
            # f_k is kept from the previous pass instead of recomputed.
            f_kp1 = getObjFVal(x[:, k], functionID)
            if k > 0 and criterion(f_k, f_kp1, grad[:,k-1], grad[:,k],Epsilon):
                if verbose:
                    print("Bingo !")
                break
            f_k = f_kp1
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k], Epsilon):
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


# H^0 of the quasi-Newton update (162)

def precond_identity(x0, n, functionID):
    return np.eye(n)


# Jacobi -> rescaling each variable by its curvature
def precond_jacobi(x0, n, functionID):
    d = np.diag(getObjFHess(x0, functionID)).astype(float).copy()
    # H^0 has to be SPD: a flat or concave direction gets the unit scaling back
    bad = ~(d > 0)
    if bad.any():
        d[bad] = 1.0
    return np.diag(1.0 / d)


# the exact inverse Hessian at x^0: BFGS then starts as Newton's method, which
# on a quadratic reaches the minimiser in a single step
def precond_inv_hessian(x0, n, functionID):
    Hess = getObjFHess(x0, functionID)
    w, V = np.linalg.eigh(Hess)
    # the Hessian of f2 is not SPD everywhere
    #so H^0 = I is used instead
    if (w <= 0).any():
        return np.eye(n)
    return (V / w) @ V.T


preconditioners = {
    "identity": precond_identity,
    "jacobi": precond_jacobi,
    "inverse hessian": precond_inv_hessian,
}


# precond is a function (or the name of one of those above) building H^0
def build_H0(precond, x0, n, functionID, verbose=True):
    build = precond if callable(precond) else preconditioners.get(precond)
    if build is None:
        raise ValueError(
            f"Unknown preconditioner {precond!r}, pick one of "
            f"{sorted(preconditioners)}"
        )
    H0 = np.asarray(build(np.asarray(x0, float), n, functionID), float)
    if H0.shape != (n, n):
        raise ValueError(
            f"preconditioner {getattr(build, '__name__', build)} returned a "
            f"{H0.shape} H^0, expected ({n}, {n})"
        )
    # -H^0 grad is a descent direction only if H^0 is SPD (151)
    if verbose:
        w = np.linalg.eigvalsh(0.5 * (H0 + H0.T))
        if not np.allclose(H0, H0.T) or w.min() <= 0:
            print(
                f"BFGS: H^0 given by "
                f"{getattr(build, '__name__', build)} is not SPD "
                f"(smallest eigenvalue {w.min():.3e}), -H^0 grad f is not "
                f"guaranteed to be a descent direction (151)",
                file=sys.stderr,
            )
    return H0


# 3) BFGS Quasi-Newton
# restart = 0 -> no re-initialization, restart = n -> the course's rule (162)
def BFGS(x, n, functionID, MaxIter, Epsilon,criterion = crit_2_norm, restart=0,
         precond=precond_identity, verbose=True):
    if verbose:
        print("You chose the BFGS Quasi-Newton method.")
        
    grad = np.zeros((n, MaxIter))

    # guess from 162, H^0 = I unless another preconditioner is given
    H0 = build_H0(precond, x[:, 0], n, functionID, verbose)
    H = H0.copy()
    Fgrad = getObjFGrad(x[:, 0], functionID)

    # should be MaxIter-1 since initialization of x= n,MaxIter
    for k in range(MaxIter - 1):
        grad[:,k] = Fgrad
        
        if criterion in NEEDS_FVAL:
            # |f(x^k) - f(x^k-1)| needs two iterates, so it cannot be evaluated
            # at k = 0: x[:, -1] is the last (still empty) column, not x^-1.
            # f_k is kept from the previous pass instead of recomputed.
            f_kp1 = getObjFVal(x[:, k], functionID)
            if k > 0 and criterion(f_k, f_kp1, grad[:,k-1], grad[:,k],Epsilon):
                if verbose:
                    print("Bingo !")
                break
            f_k = f_kp1
        else:
            if criterion(0,0, grad[:,k-1], grad[:,k],Epsilon):
                if verbose:
                    print("Bingo !")
                break
        
        
        # periodic re-initialization (162)
        # -> after n iterations, starting from x^n, go on with H = H^0
        if restart and k > 0 and k % restart == 0:
            H = H0.copy()

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
        else:
            # delta . gamma > 0 is exactly what keeps H SPD (161). it can only
            # fail when the line search returns a step that does not satisfy
            # the curvature condition, which f2 allows since it is not convex.
            # updating anyway could make -H grad an ASCENT direction, so the
            # update is skipped and H is kept as it is
            if verbose:
                print(
                    f"BFGS: k = {k}, delta . gamma = {dg:.3e} <= 0, the SPD of "
                    f"H^{k + 1} is not assured (161), update skipped",
                    file=sys.stderr,
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
