import numpy as np
from getObjFVal import *
from optimization import *
import time
import matplotlib.pyplot as plt

lb, up = -15, 15  # the domain of the statement
N = 8  # N x N grid of starting points in [lb, up]^2, to get a spread


def crit_arret(n, functionID, MaxIter, method, eps, crit) :
    
    # each metric is kept as (median, min, max) over the grid of starting points
    nb_appel = np.full((3, len(eps)), np.nan)
    nb_it = np.full((3, len(eps)), np.nan)
    time_it = np.full((3, len(eps)), np.nan)
    
    xi = np.linspace(lb, up, N)
    
    for i in range(len(eps)) :
        
        calls, iters, times, failed = [], [], [], 0
        
        for x0 in ((a, b) for a in xi for b in xi) :
            
            # a fresh x for every run: the methods write into the array they are
            # given and return a view of it, so reusing one leaves stale columns
            x = np.zeros((n, MaxIter))
            x[:, 0] = x0
            reset_counters()
            
            start = time.perf_counter()
            try:
                if method == 1:
                    x_rep = steepest_descent(x, n, functionID, MaxIter, eps[i], crit,verbose=False)
                    
                elif method == 2:
                    x_rep = conjugate_gradients_Fletcher_Reeves(x, n, functionID, MaxIter, eps[i], crit,verbose=False)
                    
                elif method == 3:
                    x_rep = BFGS(x, n, functionID, MaxIter, eps[i], crit,verbose=False)
            except MaxIterError:
                failed += 1
                continue
            end = time.perf_counter()
            
            # getObjFVal is never called by crit_1/crit_2 nor by the dichotomy,
            # so the gradient count is the one that carries the cost
            calls.append(getObjFGrad.calls)
            iters.append(x_rep.shape[1])
            times.append(end - start)
        
        # the point is kept only when every starting point converged, otherwise
        # the statistics would be taken on an easier problem than the others
        if failed or not calls:
            print(f"   {crit.__name__}: eps={eps[i]:g} -> {failed}/{N * N} "
                  f"hit MaxIter, point dropped")
            continue
        
        for arr, vals in ((nb_appel, calls), (nb_it, iters), (time_it, times)):
            v = np.array(vals, float)
            arr[:, i] = (np.median(v), v.min(), v.max())
    
    return nb_appel, nb_it, time_it


def plot_eps(n, functionID, MaxIter, eps, method):
    
    # the criterion is written out: criterion 2 compares the SQUARE of the
    # norm to the same Epsilon as the other two (21), which the formula shows
    criteria = ((crit_1_max, r'$\max_i |\nabla f_i| < \epsilon$', 'm', 'o'),
                (crit_2_norm, r'$\nabla f^T \nabla f < \epsilon$', 'darkblue', '^'),
                (crit_3_diff, r'$|f^{k+1} - f^k| < \epsilon$', 'orange', 's'))
    
    results = [crit_arret(n, functionID, MaxIter, method, eps, c)
               for c, *_ in criteria]
    styles = [(label, color, marker) for _, label, color, marker in criteria]
    
    for k, (ylabel, filename) in enumerate((
        (r'$\nabla F$ calls [-]', 'img/nb_calls_method.pdf'),
        ('Number of iterations [-]', 'img/nb_it_method.pdf'),
        ('Time [s]', 'img/time_method.pdf'),
    )):
        # a figure of its own: without it both plots land on the same axes and
        # the second file holds all six curves
        plt.figure()
        for res, (label, color, marker) in zip(results, styles):
            med = res[k][0]  # crit_arret also returns min and max, not drawn
            plt.plot(eps, med, color=color, label=label, marker=marker)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Tolerance $\epsilon$ [-]', fontsize = 13)
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
    
    functionID = 1
    method = 1
    eps = [10**-i for i in range(1,10)]
    #eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5 , 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]

    MaxIter = 100  # Maximum number of iterations
    n = 2  # Dimension of the problem
    # the starting point is no longer a single one: crit_arret sweeps the
    # N x N grid of [lb, up]^2 so that a median and a spread can be reported
    
    
    #crit_arret(x, n, functionID, MaxIter, method, eps, crit_2_norm)
    
    plot_eps(n, functionID, MaxIter, eps, method)




























