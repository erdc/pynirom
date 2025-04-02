#! /usr/bin/env python

"""
Various utility codes for visualizing
different aspects of the DMD-based
non-intrusive reduced order model
"""

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from numpy.lib.scimath import sqrt as csqrt
from scipy import interpolate


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, ScalarFormatter, FormatStrFormatter
import itertools
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
import matplotlib.ticker as ticker
from IPython.display import display
from matplotlib import animation
matplotlib.rc('animation', html='html5')

# colors = itertools.cycle()
markers = itertools.cycle(['p', 'd', 'o', '^', 's', 'x', 'D', 'H', 'v', '*'])

# Plot parameters
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 20,
                     'lines.linewidth': 2,
                     'axes.labelsize': 16,
                     'axes.titlesize': 18,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 16,
                     'axes.linewidth': 2})


def plot_NODE_err(err, tseries, soln_names, var_string,**kwargs):
    """
    Plot error metrics for NODE solution
    (Assumes a three component model: u,v,p )
    Input::
    err: Dictionary with 1d arrays of solution component error values as keys
    tseries: Scaled time series used as x-axis of plots
    soln_names: Dictionary with names of solution components
    var_string: Dictionary with labels of solution components
    mark: [Optional] Index marking end point of training window
                    in 'tseries' array
    unit: [Optional] Unit of tseries to be used in ylabel, given as a string
    metric: [Optional] Type of error metric reported in 'err' array
            Default is 'rms',
            metric = 'rel' implies relative error
    """

    ky1 = soln_names[0]; ky2 = soln_names[1]; ky3 = soln_names[2]
    if 'unit' in kwargs:
        t_unit = kwargs['unit']
    else:
        t_unit = 'seconds'

    freq = tseries.size//20

    fig = plt.figure(figsize=(16,4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(tseries[:], err[ky1][:], 'r-s', markersize=8,
                    label='$\mathbf{%s}$'%(var_string[ky1]),lw=2, markevery=freq)
    ymax_ax1 = err[ky1][:].max()
    ax1.set_xlabel('Time (%s)'%t_unit);lg=plt.legend(ncol=2, fancybox=True,)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(tseries[:], err[ky2][:], 'b-o', markersize=8,
                    label='$\mathbf{%s}$'%(var_string[ky2]), lw=2, markevery=freq)
    ax2.plot(tseries[:], err[ky3][:], 'g-^', markersize=8,
                    label='$\mathbf{%s}$'%(var_string[ky3]), lw=2, markevery=freq-10)
    ymax_ax2 = np.maximum(err[ky2][:].max(), err[ky3][:].max())
    ax2.set_xlabel('Time (%s)'%t_unit);lg=plt.legend(ncol=2, fancybox=True,)

    if 'mark' in kwargs:
        tr_mark = kwargs['mark']
        ax1.vlines(tseries[tr_mark], 0, ymax_ax1, colors ='k', linestyles='dashdot')
        ax2.vlines(tseries[tr_mark],0,ymax_ax2, colors = 'k', linestyles ='dashdot')

    if 'metric' in kwargs:
        if kwargs['metric'] == 'rel':
            fig.suptitle('Relative errors of PODNODE NIROM solutions', fontsize=18)
        elif kwargs['metric'] == 'rms':
            fig.suptitle('Spatial RMS errors of PODNODE NIROM solutions', fontsize=18)
    else:
        fig.suptitle('Spatial RMS errors of PODNODE NIROM solutions', fontsize=18)
