import numpy as np
from getObjFVal import *
from optimization import *
import time
import matplotlib.pyplot as plt

lb, up = -15, 15  # the domain of the statement
N = 30  # N x N grid of starting points in [lb, up]^2, to get a spread



def fct_tol(n, functionID, MaxIter, eps, method) :
    
    x_rep_c1 = np.zeros((n, len(eps)))
    x_rep_c2 = np.zeros((n, len(eps)))
    x_rep_c3 = np.zeros((n, len(eps)))
    fct_c1 = np.zeros(len(eps))
    fct_c2 = np.zeros(len(eps))
    fct_c3 = np.zeros(len(eps))
    
    for i in range(len(eps)) :
        xinit = np.array([0,0])
        xinit = xinit.reshape(2, 1)
        x = np.zeros((n, MaxIter))
        x[:, 0] = xinit[:, 0]  
        
        if method == 1 :

            x_rep_c1[:,i] = steepest_descent(x, n, functionID, MaxIter, eps[i], crit_1_max,verbose=False)[:,-1]
            fct_c1[i] = getObjFVal(x_rep_c1[:,i], functionID)
            
            x_rep_c2[:,i] = steepest_descent(x, n, functionID, MaxIter, eps[i], crit_2_norm,verbose=False)[:,-1]
            fct_c2[i] = getObjFVal(x_rep_c2[:,i], functionID)
            
            x_rep_c3[:,i] = steepest_descent(x, n, functionID, MaxIter, eps[i], crit_3_diff,verbose=False)[:,-1]
            fct_c3[i] = getObjFVal(x_rep_c3[:,i], functionID)
            
          
        if method == 2 :

            x_rep_c1[:,i] = conjugate_gradients_Fletcher_Reeves(x, n, functionID, MaxIter, eps[i], crit_1_max,verbose=False)[:,-1]
            fct_c1[i] = getObjFVal(x_rep_c1[:,i], functionID)
            
            x_rep_c2[:,i] = conjugate_gradients_Fletcher_Reeves(x, n, functionID, MaxIter, eps[i], crit_2_norm,verbose=False)[:,-1]
            fct_c2[i] = getObjFVal(x_rep_c2[:,i], functionID)
            
            x_rep_c3[:,i] = conjugate_gradients_Fletcher_Reeves(x, n, functionID, MaxIter, eps[i], crit_3_diff,verbose=False)[:,-1]
            fct_c3[i] = getObjFVal(x_rep_c3[:,i], functionID)
            
        
        if method == 3 :

            x_rep_c1[:,i] = BFGS(x, n, functionID, MaxIter, eps[i], crit_1_max,verbose=False)[:,-1]
            fct_c1[i] = getObjFVal(x_rep_c1[:,i], functionID)
            
            x_rep_c2[:,i] = BFGS(x, n, functionID, MaxIter, eps[i], crit_2_norm,verbose=False)[:,-1]
            fct_c2[i] = getObjFVal(x_rep_c2[:,i], functionID)
            
            x_rep_c3[:,i] = BFGS(x, n, functionID, MaxIter, eps[i], crit_3_diff,verbose=False)[:,-1]
            fct_c3[i] = getObjFVal(x_rep_c3[:,i], functionID)
            
        
        
    plt.rcParams['axes.formatter.useoffset'] = False
    plt.loglog(eps, np.abs(fct_c1) , color = 'm', label = r'$\max_i |\nabla f_i| < \epsilon$', marker ='o')
    plt.loglog(eps, np.abs(fct_c2) , color = 'darkblue', label =r'$\nabla f^T \nabla f < \epsilon$', marker = 's')
    plt.loglog(eps, np.abs(fct_c3) , color = 'orange', label =r'$|f^{k+1} - f^k| < \epsilon$', marker = '^')
    #plt.loglog(eps, np.abs(fct_c23), color = 'c' , label = 'Crit 2 and Crit 3', marker = 'x')
    plt.xlabel('Tolerance eps [-]', fontsize = 13)
    plt.ylabel('Function value [-]', fontsize = 13)
    plt.grid()
    plt.legend()
    plt.gca().invert_xaxis() 
    #plt.savefig('bfgs_fct2_tol.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    
    return 

#def crit_2_3_comb(f_k, f_kp1, grad_k, grad_kp1,Epsilon):
    #return crit_2_norm(f_k, f_kp1, grad_k, grad_kp1,1e-5) and crit_3_diff(f_k, f_kp1, grad_k, grad_kp1,Epsilon)



def crit_comb(n, functionID, MaxIter, eps) :
    
    #x_rep_c1 = np.zeros((n, len(eps)))
    #x_rep_c2 = np.zeros((n, len(eps)))
    #x_rep_c3 = np.zeros((n, len(eps)))
    fct_c1 = np.zeros(len(eps))
    fct_c2 = np.zeros(len(eps))
    fct_c3 = np.zeros(len(eps))
    
    nb_it_1 = np.zeros(len(eps))
    nb_it_2 = np.zeros(len(eps))
    nb_it_3 = np.zeros(len(eps))
    #tol_norm2 = 1e-5
    
    Epsilon = 1e-5
    
    #to compare, norm 2 alone
    xinit = np.array([0,0])
    xinit = xinit.reshape(2, 1)
    x = np.zeros((n, MaxIter))
    x[:, 0] = xinit[:, 0]
    
    
    x_rep_m1 = steepest_descent(x.copy(), n, functionID, MaxIter, Epsilon, crit_2_norm,verbose=False)
    fct_m1 = getObjFVal(x_rep_m1[:,-1], functionID)
    it_m1 = x_rep_m1.shape[1]
    
    
    x_rep_m2= conjugate_gradients_Fletcher_Reeves(x.copy(), n, functionID, MaxIter, Epsilon, crit_2_norm,verbose=False)
    fct_m2 = getObjFVal(x_rep_m2[:,-1], functionID)
    it_m2 = x_rep_m2.shape[1]
    
    x_rep_m3 = BFGS(x.copy(), n, functionID, MaxIter, Epsilon, crit_2_norm,verbose=False)
    fct_m3 = getObjFVal(x_rep_m3[:,-1], functionID)
    it_m3 = x_rep_m3.shape[1]
    
    
    #analysis of convergence
    for i in range(len(eps)) :
        
        xinit = np.array([0,0])
        xinit = xinit.reshape(2, 1)
        x = np.zeros((n, MaxIter))
        x[:, 0] = xinit[:, 0]  
    
            
        x_rep_c1 = steepest_descent(x, n, functionID, MaxIter, eps[i], crit_2_3_comb,verbose=False)
        fct_c1[i] = getObjFVal(x_rep_c1[:,-1], functionID)
        
        x_rep_c2 = conjugate_gradients_Fletcher_Reeves(x, n, functionID, MaxIter, eps[i], crit_2_3_comb,verbose=False)
        fct_c2[i] = getObjFVal(x_rep_c2[:,-1], functionID)
        
        x_rep_c3 = BFGS(x, n, functionID, MaxIter, eps[i], crit_2_3_comb,verbose=False)
        fct_c3[i] = getObjFVal(x_rep_c3[:,-1], functionID)
        
        
        nb_it_1[i] = x_rep_c1.shape[1]
        nb_it_2[i] = x_rep_c2.shape[1]
        nb_it_3[i] = x_rep_c3.shape[1]
    
            
        
        
    
    plt.rcParams['axes.formatter.useoffset'] = False
    plt.axhline(fct_m1, color = 'lightgreen', label ='Steepest Descent - norm2')
    plt.axhline(fct_m2, color = 'lightblue', label ='Conj. Grad. - norm2')
    plt.axhline(fct_m3, color = 'lightcoral', label ='BFGS - norm2')
    plt.semilogx(eps, fct_c1 , color = 'darkgreen', label = 'Steepest Descent - comb', marker ='o')
    plt.semilogx(eps, fct_c2 , color = 'blue', label ='Conj. Grad. - comb', marker = 's')
    plt.semilogx(eps, fct_c3 , color = 'red', label ='BFGS - comb', marker = '^')
    plt.xlabel('Tolerance eps of crit 3 [-]', fontsize = 13)
    plt.ylabel('Function value [-]', fontsize = 13)
    plt.grid()
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.savefig('comb_fct2_val.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    
    plt.rcParams['axes.formatter.useoffset'] = False
    plt.axhline(it_m1, color = 'lightgreen', label ='Steepest Descent - norm2')
    plt.axhline(it_m2, color = 'lightblue', label ='Conj. Grad. - norm2')
    plt.axhline(it_m3, color = 'lightcoral', label ='BFGS - norm2')
    plt.loglog(eps, nb_it_1 , color = 'darkgreen', label = 'Steepest Descent - comb', marker ='o')
    plt.loglog(eps, nb_it_2 , color = 'blue', label ='Conj. Grad. - comb', marker = 's')
    plt.loglog(eps, nb_it_3 , color = 'red', label ='BFGS - comb', marker = '^')
    plt.xlabel('Tolerance eps of crit 3 [-]', fontsize = 13)
    plt.ylabel('Number of iterations [-]', fontsize = 13)
    plt.grid()
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.savefig('comb_fct2_it.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    
    
    
    
    return nb_it_1, nb_it_2, nb_it_3




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
            
            print(x_rep.shape[1])
            
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
        (r'$\nabla F$ calls [-]', 'img1/2_bfgs_nb_calls.pdf'),
        ('Number of iterations [-]', 'img1/2_bfgs_nb_it.pdf'),
        ('Time [s]', 'img1/2_bfgs_time.pdf'),
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




if __name__== '__main__':
    
    functionID = 2
    method = 3
    eps = [10**-i for i in range(1,10)]
    #eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5 , 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]

    MaxIter = 100  # Maximum number of iterations
    n = 2  # Dimension of the problem
    # the starting point is no longer a single one: crit_arret sweeps the
    # N x N grid of [lb, up]^2 so that a median and a spread can be reported
    #crit = crit_3_diff
    
    xinit = np.array([0,0])
    xinit = xinit.reshape(2, 1)
    x = np.zeros((n, MaxIter))
    x[:, 0] = xinit[:, 0]  
    
    
    crit_comb(n, functionID, MaxIter, eps)
    #crit_arret(n, functionID, MaxIter, method, eps, crit_2_norm)
    #fct_tol(n, functionID, MaxIter, eps, method)
    #plot_eps(n, functionID, MaxIter, eps, method)




























