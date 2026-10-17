import numpy as np
from getObjFVal import getObjFVal, getObjFGrad, getObjFHess


# Line search method 1: dichotomy with derivatives (125)
# how should we set h ?
def dichotomy(
    x,
    s,
    functionID,
    tol=1e-5,
    h=0.1,
    rho=0.5,
):
    # Dichotomy method ( 125)
    # rho = 0.5 -> bisection !
    def dphi(alpha):
        return getObjFGrad(x + alpha * s, functionID) @ s

    amin, amax = 0.0, h
    # determine the first interval (127)
    while dphi(amax) < 0:
        amin, amax = amax, 2 * amax
    # each iteration (126)
    while amax - amin > tol:
        a = amax - rho * (amax - amin)
        amin, amax = (a, amax) if dphi(a) < 0 else (amin, a)
    a = amax - rho * (amax - amin)
    # phi not necessary unimodal, shrink required (140)
    # f2 is not convex
    while getObjFVal(x + a * s, functionID) >= getObjFVal(x, functionID) and a > tol:
        a *= rho
    return a


# Line search method 2: Newton-Raphson (117)


# h and rho are ignored, the method starts from alpha = 0 as in the course
def newton_raphson(x, s, functionID, tol=1e-5, h=0.1, rho=0.5, alpha=0.0):

    def phi(alpha):
        return getObjFVal(x + alpha * s, functionID)

    def dphi(alpha):
        return getObjFGrad(x + alpha * s, functionID) @ s

    def ddphi(alpha):
        return s.T @ getObjFHess(x + alpha * s, functionID) @ s

    while abs(dphi(alpha)) > tol:
        alpha -= dphi(alpha) / ddphi(alpha)

    return alpha


def quadratic(x, s, functionID, tol=1e-5, h=0.1, rho=0.5, MaxIter=100):

    def phi(alpha):
        return getObjFVal(x + alpha * s, functionID)

    # choose an initial value a_0 and an arbitrary step size delta
    # set a_1 = a_0 + delta and compute phi(a_0) and phi(a_1)
    a_0, delta = 0.0, h

    # SITUATION 2 phi(a_1) > phi(a_0)
    # -> a_0 is now a_1
    # -> delta *= -1
    # and proceed as above
    if phi(a_0 + delta) > phi(a_0):
        a_0, delta = a_0 + delta, -delta

    # SITUATION 1 phi(a_1) <= phi(a_0)
    # a_k = a_k-1 + 2^{k-1} delta
    # loop until phi(a_k) increase
    alphas, phis = [a_0, a_0 + delta], [phi(a_0), phi(a_0 + delta)]
    while phis[-1] <= phis[-2] and len(alphas) <= MaxIter:
        alphas.append(alphas[-1] + 2 ** (len(alphas) - 1) * delta)
        phis.append(phi(alphas[-1]))
    # phi(a_0) >= phi(a_1) >= ... >= phi(a_p-1) BUT phi(a_p-1) < phi(a_p)
    p = len(alphas) - 1
    if phis[p] <= phis[p - 1]:
        return alphas[p]  # phi still decreasing, s looks unbounded below

    # a_p+1 = a_p - 2^{p-2} delta
    # phi(a_p+1)
    alpha_p1 = alphas[p] - 2 ** (p - 2) * delta
    phi_p1 = phi(alpha_p1)

    # -> 4 points obtained, separated by the same distance Delta = 2^{p-2} delta
    # a_p-2 < a_p-1 < a_p+1 < a_p
    # a, b, c are obtained by eliminating (133)
    # -> a_p   if phi(a_p-1) < phi(a_p+1)
    # -> a_p-2 if phi(a_p+1) < phi(a_p-1)
    if phis[p - 1] < phi_p1:
        a1, a2, a3 = alphas[p - 2], alphas[p - 1], alpha_p1
        phi1, phi2, phi3 = phis[p - 2], phis[p - 1], phi_p1
    else:
        a1, a2, a3 = alphas[p - 1], alpha_p1, alphas[p]
        phi1, phi2, phi3 = phis[p - 1], phi_p1, phis[p]
    # in situation 2 delta < 0, the walk runs backwards -> reorder into a < b < c
    if a1 > a3:
        a1, a3, phi1, phi3 = a3, a1, phi3, phi1

    # PRINCIPLE (128)
    # a_1 < a_2 < a_3 with phi(a_1) >= phi(a_2) and phi(a_3) >= phi(a_2)
    # -> approximate phi on [a_1, a_3] by the quadric q through the 3 points
    for _ in range(MaxIter):
        if a3 - a1 < tol:
            break

        # big formula from 129
        r23, r31, r12 = a2**2 - a3**2, a3**2 - a1**2, a1**2 - a2**2
        s23, s31, s12 = a2 - a3, a3 - a1, a1 - a2
        den = s23 * phi1 + s31 * phi2 + s12 * phi3
        a4 = a2 if den == 0 else 0.5 * (r23 * phi1 + r31 * phi2 + r12 * phi3) / den

        # phi not necessary unimodal, q can degenerate into a line (den = 0) or
        # push a_4 out of ]a_1, a_3[ -> bisect the widest half instead
        if a4 == a2 or not a1 < a4 < a3:
            a4 = 0.5 * (a2 + a3) if a3 - a2 > a2 - a1 else 0.5 * (a1 + a2)
        phi4 = phi(a4)

        #big if else slide 129
        if phi4 <= phi2:
            if a4 > a2:
                a1, phi1 = a2, phi2
            else:
                a3, phi3 = a2, phi2
            a2, phi2 = a4, phi4
        else:
            if a4 > a2:
                a3, phi3 = a4, phi4
            else:
                a1, phi1 = a4, phi4

    # a_4 is taken as an approximate of the optimum of phi on [a_1, a_3] (129)
    return a2


linear_methods = {
    "dichotomy": dichotomy,
    "newton-raphson": newton_raphson,
    "quadratic": quadratic,
}


def linear_search(x, s, functionID, tol, h=0.1, rho=0.5, method="newton-raphson"):
    # fancy way to call the function of the name of method
    search = method if callable(method) else linear_methods.get(method)
    if search is None:
        raise ValueError(
            f"Unknown line search {method!r}, pick one of {sorted(linear_methods)}"
        )
    return search(x, s, functionID, tol, h, rho)
