import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from getObjFVal import *
from optimization import *

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 13,
    "mathtext.fontset": "dejavuserif",
    
})


styles = {
    "Steepest descent":    dict(color="C0", linestyle="-",  marker="o"),
    "Conjugate gradients": dict(color="C1", linestyle="--", marker="s"),
    "BFGS":                dict(color="C2", linestyle=":",  marker="^"),
}

MaxIter = 300
Epsilon = 1e-5
functionID = 1

if functionID == 1:
    START_POINTS = {
        "Steepest descent":    (-2.5, -14),
        "Conjugate gradients": (-2.5, -14),
        "BFGS":                (-2.5, -14),
    }

elif functionID == 2:
    START_POINTS = {
        "Steepest descent":    (-1.5, 12.5),
        "Conjugate gradients": (-1.5, 12.5),
        "BFGS":                (-1.5, 12.5),
    }


def getFValHist(name):
    
    x = np.zeros((2, MaxIter))
    x[:, 0] = START_POINTS[name]
    
    x = methods[name](x, 2, functionID, MaxIter, Epsilon, verbose=False)
    
    nit = x.shape[1]
    hist = getObjFVal(x, functionID)
    
    return nit, hist


if __name__ == "__main__":
    
    fig, ax = plt.subplots()
    
    for name in methods:
        
        if functionID == 1:
            f_star = -8.022727272727
        
        elif functionID == 2:
            f_star = -23.324709104004
        
        cut = None
    
        nit, hist = getFValHist(name)
        
        it = np.arange(1, nit+1, 1)
        
        ax.set_xlabel("Iterations")
        ax.set_ylabel(rf"$|f_{{{functionID}}} - f_{{{functionID},\infty}}|$")
        ax.plot(it-1, np.abs(hist - f_star), **styles[name], label = name)
        ax.grid()
        ax.legend()
        
        if cut != None:
            ax.set_xlim(right=cut)
            
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax.xaxis.set_major_locator(MultipleLocator(1))
        
    plt.savefig(f"img/{functionID}_conv.pdf", dpi=300, bbox_inches="tight")
    plt.show()
        
