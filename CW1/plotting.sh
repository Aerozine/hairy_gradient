#!/bin/sh
# regenerate every figure of img/ in one go.
# Agg: the scripts that end on plt.show() would otherwise block the next one.
set -e
export MPLBACKEND=Agg

for script in \
    plot_sensi.py \
    plot-twist.py \
    fun_calls_plot.py \
    plot_reinit.py \
    rate_of_convergence.py \
    plot_precond.py \
    plot_linesearch.py \
    plot_hrho.py
do
    echo "--> $script"
    python "$script"
done

echo "done, figures are in img/"
