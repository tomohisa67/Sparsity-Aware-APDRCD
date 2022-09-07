# -*- coding: utf-8 -*-
"""
APDRCD without duplicate sampling for entropy regularized ot
"""
# Author: T.Tabuchi
# Date  : 2022/9/7
# References: https://github.com/PythonOT/POT.git

import numpy as np

from algorithms.utils import compute_matrixH, shuffle, tic, toc

def pdrcd(r, l, C, reg, maxIter, err_flag=1, time_flag=1, value_flag=1):
    r'''
    Compute the APDRCD algorithm to solve the regularized discrete measures optimal transport problem

    Parameters
    ----------
    r : ndarray, shape (ns,),
        Source measure.
    l : ndarray, shape (nt,),
        Target measure.
    C : ndarray, shape (ns, nt),
        Cost matrix.
    reg : float
          Regularization term > 0
    maxIter : int
        Number of iteration.
    Returns
    -------
    x_tilde : ndarray, shape (ns, nt),
              Optimal Solution (Transport matrix)
    Examples
    --------
    References
    --------
    '''

    # parameters
    theta = 1.0
    Ck = 1.0
    L = 4/reg

    # set up
    ns = r.shape[0]
    nt = l.shape[0]
    b = np.concatenate((r,l))
    c = C.flatten()
    H = compute_matrixH(ns,nt)

    # primal (dual) variables
    x = np.zeros(ns*nt)
    y = np.zeros(ns+nt)

    # output
    x_tilde = np.zeros(ns*nt)
    err_list = []
    time_list = []
    value_list = []

    # sampling strategy
    sampling_list = shuffle(ns,nt,maxIter)

    if time_flag != 0:
        tic()
    # main loop
    for k in range(maxIter):
        x = np.exp((-c-H.T@y)/reg)

        j = sampling_list[k]
        j = int(j)

        y[j] = y[j] - 1/L*(b[j]-H[j]@x)

        if err_flag != 0:
            err = np.linalg.norm(H@x-b, ord=1)
            err_list.append(err)

        if time_flag != 0:
            t = toc()
            time_list.append(t)

        if value_flag != 0:
            value = c.T@x
            value_list.append(value)

    return x, err_list, time_list, value_list