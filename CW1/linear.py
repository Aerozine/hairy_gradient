import numpy as np
from getObjFVal import getObjFVal, getObjFGrad, getObjFHess


#
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
    # to be removed
    while getObjFVal(x + a * s, functionID) >= getObjFVal(x, functionID) and a > tol:
        a *= rho
    return a


# Line search method 2: Newton-Raphson (117)


# x, s and alpha are current iterates i in the method calling the function
def newton_raphson(x, s, alpha, functionID, tol):

    def phi(alpha):
        return getObjFVal(x + alpha * s, functionID)

    def dphi(alpha):
        return getObjFGrad(x + alpha * s, functionID) @ s

    def ddphi(alpha):
        return s.T @ getObjFHess(x + alpha * s, functionID) @ s

    while abs(dphi(alpha)) > tol:
        alpha -= dphi(alpha) / ddphi(alpha)

    return alpha


# simple selector for the linear search method ( mainly used for plotting )
# todo using enum or sth instead of NR= T/F
def linear_search(x, s, functionID, tol, h=0.1, rho=0.5, NR=False):
    if NR:
        return newton_raphson(x, s, 0.0, functionID, tol)
    return dichotomy(x, s, functionID, tol, h, rho)
