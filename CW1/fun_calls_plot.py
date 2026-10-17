#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from getObjFVal import *
from optimization import *

plt.rcParams.update({
"font.family": "sans-serif", 
"font.sans-serif": "Tahoma",
"mathtext.fontset": "dejavuserif",
"xtick.labelsize": 30,  
"ytick.labelsize": 30,
"xtick.major.pad": 20,
"ytick.major.pad": 20,
'font.size': 35,
"figure.figsize": (15, 12),
"lines.linewidth": 3,
"axes.labelpad": 15,
"savefig.format": 'pdf',
"grid.linewidth": 1,
"xtick.major.size": 10,
"ytick.major.size": 10,
"legend.fontsize": 35,
"lines.markersize": 10
})


styles = {
    "Steepest descent":    dict(color="C0", linestyle="-",  marker="o"),
    "Conjugate gradients": dict(color="C1", linestyle="--", marker="s"),
    "BFGS":                dict(color="C2", linestyle=":",  marker="^"),
}

MaxIter = 300
Epsilon = 1e-5
x0 = np.array([0,0])
functionID = 2


def getFValHist(name):
    
    x = np.zeros((2, MaxIter))
    x[:, 0] = x0
    
    x = methods[name](x, 2, functionID, MaxIter, Epsilon, verbose=False)
    
    nit = x.shape[1]
    hist = getObjFVal(x, functionID)
    
    return nit, hist


if __name__ == "__main__":
    
    fig, ax = plt.subplots()
    
    for name in methods:
        
        cut = None
    
        nit, hist = getFValHist(name)
        
        it = np.arange(1, nit+1, 1)

        ax.plot(it, hist, **styles[name], label = name)
        ax.grid()
        ax.legend()
        
        if cut != None:
            ax.set_xlim(right=cut)
            
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.show()
        
