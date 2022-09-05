# -*- coding: utf-8 -*-
"""
Sparsity-Aware APDRCD without duplicate sampling for entropy regularized ot
"""
# Author: T.Tabuchi
# Date  : 2022/9/5
# References: https://github.com/PythonOT/POT.git

import numpy as np

from algorithms.utils import creat_nonzerorow_index, compute_matrixH, shuffle, tic, toc

def sa_apdrcd(r, l, C, reg, maxIter=4000, err_flag=1, time_flag=1, value_flag=1):
    r'''
    Compute the SA-APDRCD algorithm to solve the regularized discrete measures optimal transport problem

    Parameters
    ----------
    a : ndarray, shape (ns,),
        Source measure.
    b : ndarray, shape (nt,),
        Target measure.
    C : ndarray, shape (ns, nt),
        Cost matrix.
    reg : float
        Regularization term > 0
    maxIter : int
        Number of iteration.
    Returns
    -------
    Examples
    --------
    References
    --------
    '''

    # parameter
    theta = 1
    Ck = 1
    L = 4/reg

    # set up
    ns = r.shape[0]
    nt = l.shape[0]
    b = np.concatenate((r,l))
    c = C.flatten()
    H = compute_matrixH(ns,nt)

    # non-zero index
    J, I = creat_nonzerorow_index(H,ns.nt)

    # sampling strategy
    sampling_list = shuffle(ns,nt,maxIter)

    # primal (dual) variables
    x = np.zeros(ns*nt)
    y = np.zeros(ns+nt)

    # auxiliary dual variables
    u = np.zeros(ns+nt)
    v = np.zeros(ns+nt)

    # output
    x_tilde = np.zeros(ns*nt)
    err_list = []
    time_list = []
    value_list = []

    if time_flag != 0:
        tic()
    # main loop
    for k in range(maxIter):
        j = sampling_list[k]

        # Hx = H[j]@x
        u[j] = y[j] - 1/L*(H[j]@x-b[j])
        v[j] = y[j] - 1/((ns+nt)*L*theta)*(H[j]@x-b[j])

        y[j] = (1-theta)*u[j] + theta*v[j]

        for i in I[j]:
            x[i] = np.exp((-c[i]-H.T[i]@y)/reg)
            # or #
            # i1,i2 = J[i]
            # x[i] = np.exp((-c[i]-y[i1]-y[i2])/reg)

        Ck = Ck + 1/theta
        theta = -theta**2 + np.sqrt(theta**4 + 4*theta**2)
        x_tilde = 1/Ck*(x_tilde + x/theta)

        if err_flag != 0:
            err = np.linalg.norm(H@x-b, ord=1)
            err_list.append(err)

        if time_flag != 0:
            t = toc()
            time_list.append(t)

        if value_flag != 0:
            v = c.T@x
            value_list.append(v)

    return x_tilde, err_list, time_list, value_list