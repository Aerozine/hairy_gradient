import functools
import numpy as np
from getObjFVal import *
from optimization import *
import linear
import optimization
import time
import matplotlib.pyplot as plt

optimization.linear_search = functools.partial(linear.linear_search)


def crit_arret(x, n, functionID, MaxIter, method, eps, crit) : 
    
    x0 = x[:, 0].copy()
    nb_appel = np.full(len(eps), np.nan)
    time_it = np.full(len(eps), np.nan)
    
    for i in range(len(eps)) :
        
        # a fresh x for every run: the methods write into the array they are
        # given and return a view of it, so reusing one leaves stale columns
        x = np.zeros((n, MaxIter))
        x[:, 0] = x0
        reset_counters()
        
        start = time.perf_counter()
        try:
            if method == 1:
                steepest_descent(x, n, functionID, MaxIter, eps[i], crit,verbose=False)
                
            elif method == 2:
                conjugate_gradients_Fletcher_Reeves(x, n, functionID, MaxIter, eps[i], crit,verbose=False)
                
            elif method == 3:
                BFGS(x, n, functionID, MaxIter, eps[i], crit,verbose=False)
        except MaxIterError:
            # no convergence for this eps -> left out of the curve
            print(f"   {crit.__name__}: eps={eps[i]:g} hit MaxIter, point dropped")
            continue
        end = time.perf_counter()
        
        # getObjFVal is never called by crit_1/crit_2 nor by the dichotomy, so
        # the gradient count is the one that carries the cost
        nb_appel[i] = getObjFGrad.calls
        time_it[i] = end - start
    
    return nb_appel, time_it


def plot_eps(x, n, functionID, MaxIter, eps, method):
    
    nb_appel_max, time_it_max = crit_arret(x, n, functionID, MaxIter, method, eps, crit_1_max)
    nb_appel_norm, time_it_norm = crit_arret(x, n, functionID, MaxIter, method, eps, crit_2_norm)
    nb_appel_diff, time_it_diff = crit_arret(x, n, functionID, MaxIter, method, eps, crit_3_diff)
    
    styles = (('Criterion 1 (max)', 'm', 'o'),
              ('Criterion 2 (norm)', 'darkblue', '^'),
              ('Criterion 3 (diff)', 'orange', 's'))
    
    for values, ylabel, filename in (
        ((nb_appel_max, nb_appel_norm, nb_appel_diff),
         'getObjFGrad calls [-]', 'img/nb_it_method.pdf'),
        ((time_it_max, time_it_norm, time_it_diff),
         'Time [s]', 'img/time_method.pdf'),
    ):
        # a figure of its own: without it both plots land on the same axes and
        # the second file holds all six curves
        plt.figure()
        for v, (label, color, marker) in zip(values, styles):
            plt.loglog(eps, v, color=color, label=label, marker=marker)
        plt.xlabel('Tolerance eps [-]', fontsize = 13)
        plt.ylabel(ylabel, fontsize = 13)
        plt.grid()
        plt.legend()
        plt.gca().invert_xaxis()  # once per figure, twice cancels out
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    return


'''
def plot_eps(x, n, functionID, MaxIter, eps, method):
    
    nb_appel_steep, time_it_steep = crit_arret(x, n, functionID, MaxIter, 1, eps, crit)
    nb_appel_cg, time_it_cg = crit_arret(x, n, functionID, MaxIter, 2, eps, crit)
    nb_appel_BFGS, time_it_BFGS = crit_arret(x, n, functionID, MaxIter, 3, eps, crit)
    
    
    
    #Nb iterations
    plt.loglog(eps, nb_appel_steep, color = 'm', label ='Steepest Descent')
    plt.loglog(eps, nb_appel_cg, color = 'darkblue', label = 'Conjugate gradients')
    plt.loglog(eps, nb_appel_BFGS, color = 'orange', label ='BFGS')
    plt.scatter(eps, nb_appel_steep, color = 'm', marker = 'o')
    plt.scatter(eps, nb_appel_cg, color = 'darkblue', marker = '^')
    plt.scatter(eps, nb_appel_BFGS, color = 'orange', marker = 's')
    plt.xlabel('Stopping criteria [-]', fontsize = 13)
    plt.ylabel('Number of iterations [-]', fontsize = 13)
    #plt.tick_params(axis = 'both', labelsize=12)
    #plt.xlim([100,110])
    #plt.ylim([0,300])
    plt.grid()
    plt.legend()
    plt.gca().invert_xaxis()
    #plt.savefig('nb_it_method.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    #Time
    plt.loglog(eps, time_it_steep, color = 'm', label ='Steepest Descent')
    plt.loglog(eps, time_it_cg, color = 'darkblue', label = 'Conjugate gradients')
    plt.loglog(eps, time_it_BFGS, color = 'orange', label ='BFGS')
    plt.scatter(eps, time_it_steep, color = 'm', marker = 'o')
    plt.scatter(eps, time_it_cg, color = 'darkblue', marker = '^')
    plt.scatter(eps, time_it_BFGS, color = 'orange', marker = 's')
    plt.xlabel('Stopping criteria [-]', fontsize = 13)
    plt.ylabel('Time [s]', fontsize = 13)
    plt.gca().invert_xaxis()
    #plt.tick_params(axis = 'both', labelsize=12)
    #plt.xlim([100,110])
    #plt.ylim([0,300])
    plt.grid()
    plt.legend()
    #plt.savefig('time_method.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    
    return
'''





if __name__== '__main__':
    
    functionID = 2
    method = 3
    eps = [10**-i for i in range(1,10)]
    #eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5 , 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]

    xinit = np.array([0, 0])  # initial point
    MaxIter = 100  # Maximum number of iterations
    n = 2  # Dimension of the problem
    xinit = xinit.reshape(2, 1)  # To be sure that it's a column vector
    x = np.zeros((n, MaxIter))  # Initialization of vector x
    x[:, 0] = xinit[:, 0]  # Put xinit in vector x.
    
    
    #crit_arret(x, n, functionID, MaxIter, method, eps)
    
    plot_eps(x, n, functionID, MaxIter, eps, method)




























