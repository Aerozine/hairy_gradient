import numpy as np
from getObjFVal import *
from optimization import *
from linear import linear_search
import time
import matplotlib.pyplot as plt



def crit_arret(x, n, functionID, MaxIter, method, eps, crit) : 
    
    #eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    nb_appel = np.zeros(len(eps))
    time_it = np.zeros(len(eps))
    
    for i in range(len(eps)) :
        
        start = time.perf_counter()
        if method == 1:
            steepest_descent(x, n, functionID, MaxIter, eps[i], crit)
            #print(f"Les fougeres c'est le must ! {getObjFVal.calls}")
            
        elif method == 2:
            conjugate_gradients_Fletcher_Reeves(x, n, functionID, MaxIter, eps[i], crit)
            
        elif method == 3:
            BFGS(x, n, functionID, MaxIter, eps[i], crit)
        end = time.perf_counter()
        
        nb_appel[i] = getObjFVal.calls
        reset_counters()
        time_it[i] = end - start
        
    print(f"Les fougeres c'est le must !")
    print('Eps : ', eps)
    print('Nb_appel : ', nb_appel)
    print('Time_it :', time_it)
    
    
    return nb_appel, time_it



def plot_eps(x, n, functionID, MaxIter, eps, method):
    
    nb_appel_max, time_it_max = crit_arret(x, n, functionID, MaxIter, method, eps, crit_1_max)
    nb_appel_norm, time_it_norm = crit_arret(x, n, functionID, MaxIter, method, eps, crit_2_norm)
    nb_appel_diff, time_it_diff = crit_arret(x, n, functionID, MaxIter, method, eps, crit_3_diff)
    
    
    
    #Nb iterations
    plt.loglog(eps, nb_appel_max, color = 'm', label ='Criterion 1')
    plt.loglog(eps, nb_appel_norm, color = 'darkblue', label = 'Criterion 2')
    plt.loglog(eps, nb_appel_diff, color = 'orange', label ='Criterion 3')
    plt.scatter(eps, nb_appel_max, color = 'm', marker = 'o')
    plt.scatter(eps, nb_appel_norm, color = 'darkblue', marker = '^')
    plt.scatter(eps, nb_appel_diff, color = 'orange', marker = 's')
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
    plt.loglog(eps, time_it_max, color = 'm', label ='Criterion 1')
    plt.loglog(eps, time_it_norm, color = 'darkblue', label = 'Criterion 2')
    plt.loglog(eps, time_it_diff, color = 'orange', label ='Criterion 3')
    plt.scatter(eps, time_it_max, color = 'm', marker = 'o')
    plt.scatter(eps, time_it_norm, color = 'darkblue', marker = '^')
    plt.scatter(eps, time_it_diff, color = 'orange', marker = 's')
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




























