#! /usr/bin/env python

"""
Module for utility codes for visualizing
various features of the POD-RBF
reduced order model
"""

import itertools
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
import matplotlib.ticker as ticker
from IPython.display import display
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from numpy.lib.scimath import sqrt as csqrt
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, ScalarFormatter, FormatStrFormatter

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


def plot_sing_val(D_pod):
    """
    Plot the singular value decay
    """

    fig = plt.figure(figsize=(7, 5))
    comp = D_pod.keys()
    index = {}
    mkskip = D_pod[list(D_pod.keys())[0]].shape[0]//15

    for key in D_pod.keys():
        index[key] = np.arange(D_pod[key].shape[0]-1)
        plt.semilogy(index[key], D_pod[key][:-1], marker=next(markers),
                     markevery=mkskip, markersize=8,
                     label='%s' % key, linewidth=3)

    ax = plt.gca()
    ax.set_title('Singular values of snapshot matrix')
    ax.set_ylabel('$\log$ $|\sigma|$')
    ax.set_xlabel('# of singular values')
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    plt.legend(fontsize=16)



def viz_sol(uh, nodes, triangles):
    """
    Visualize the NIROM and HFM solutions over the physical domain
    """

    boundaries = np.linspace(np.amin(uh), np.amax(uh), 11)
    cf = plt.tripcolor(nodes[:, 0], nodes[:, 1],
                       triangles, uh, cmap=plt.cm.jet, shading='gouraud')
    plt.axis('equal')

    return cf, boundaries


def viz_err(uh, snap, nodes, triangles):
    """
    Visualize the NIROM solution relative error over the domain
    """

    cf3 = plt.tripcolor(nodes[:, 0], nodes[:, 1],
                        triangles, uh-snap, cmap=plt.cm.jet, shading='flat')
    plt.axis('equal')
    cb = plt.colorbar(cf3)

    return cf3


def plot_comp_err(rms, times_online, key, lbl, clr='r', mkr='p', t_end=False, **kwargs):
    """
    Plot rms errors vs time for various reduced solutions
    """
    if 'unit' in kwargs:
        t_unit = kwargs['unit']
    else:
        t_unit = 'seconds'

    if t_end == False:
        N_end = np.count_nonzero(
            times_online[times_online <= times_online[-1]])
        index = times_online
        end_trunc = N_end+1
    else:
        N_end = np.count_nonzero(times_online[times_online < t_end])
        index = times_online[:N_end+1]
        end_trunc = N_end+1

    try:
        start_trunc = kwargs['start']
    except:
        start_trunc = 0

    mkr_skip = len(index[start_trunc:])//25
    plt.plot(index[start_trunc:], rms[key][start_trunc:end_trunc], color=clr, marker=mkr, markersize=8,
             label='$\mathbf{%s}$' % (lbl), linewidth=2, markevery=mkr_skip)
    plt.xlabel('Time (%s)'%t_unit,fontsize=16);

    if 'metric' in kwargs:
        if kwargs['metric'] == 'rel':
            plt.title('Relative errors of PODRBF NIROM solutions for $\mathbf{%s}$'%lbl, fontsize=18)
        elif kwargs['metric'] == 'rms':
            plt.title('Spatial RMS errors of PODRBF NIROM solutions for $\mathbf{%s}$'%lbl, fontsize=18)
    else:
        plt.title('Spatial RMS errors of PODRBF NIROM solutions for $\mathbf{%s}$'%lbl, fontsize=18)


def plot_RBF_err(err, tseries, soln_names, var_string,**kwargs):
    """
    Plot error metrics for RBF solution
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
            fig.suptitle('Relative errors of PODRBF NIROM solutions', fontsize=18)
        elif kwargs['metric'] == 'rms':
            fig.suptitle('Spatial RMS errors of PODRBF NIROM solutions', fontsize=18)
    else:
        fig.suptitle('Spatial RMS errors of PODRBF NIROM solutions', fontsize=18)
