#! /usr/bin/env python

import numpy as np
import scipy


"""
Simple utilities for managing snapshots and
creating training, testing data
"""

def prepare_data(data, soln_names, **options):
    """
    Utility to extract snapshots and time
    arrays from raw data, by ignoring
    initial spin-up times, skipping over
    snapshots, and setting end points.
    """
    ### Keys used in snapshot files are different
    if 'time' in data.keys():
        t_array = data['time']
    elif 'T' in data.keys():
        t_array = data['T']

    try:
        snap_start = options['start_skip']
    except:
        snap_start = 0
    try:
        snap_end = options['end_skip']
    except:
        snap_end = -1

    try:         ## Overwrites "snap_start"
        T_start = options['T_start']
        snap_start = np.count_nonzero(t_array[t_array <= T_start])
    except:
        T_start = t_array[0]
    try:         ## Overwrites "snap_end"
        T_end = options['T_end']
        snap_end = np.count_nonzero(t_array[t_array <= T_end])+1
    except:
        T_end = t_array[-1]

    try:
        incr = options['incr']
    except:
        incr = 1

    snap = {}
    for key in soln_names:
        snap[key] = data[key][:,snap_start:snap_end:incr]
    times = t_array[snap_start:snap_end:incr]

    return snap, times
