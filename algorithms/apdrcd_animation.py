from algorithms.apdrcd import apdrcd
# from algorithms.sa_apdrcd import sa_apdrcd
# from algorithms.pdrcd import pdrcd

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib

from algorithms.utils import compute_matrixH, shuffle, tic, toc

def apdrcd_animation(r, l, C, reg, maxIter, err_flag=1, time_flag=1, value_flag=1):
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

    fig, ax = plt.subplots(1, 1)

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

    # auxiliary dual variables
    u = np.zeros(ns+nt)
    v = np.zeros(ns+nt)

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
        y = (1-theta)*u + theta*v
        x = np.exp((-c-H.T@y)/reg)

        j = sampling_list[k]
        j = int(j)

        u[j] = y[j] - 1/L*(H[j]@x-b[j])
        v[j] = y[j] - 1/((ns+nt)*L*theta)*(H[j]@x-b[j])

        Ck = Ck + 1/theta
        theta = -theta**2 + np.sqrt(theta**4 + 4*theta**2)
        x_tilde = 1/Ck*(x_tilde + x/theta)

        # animation
        # if k%10 == 0:
        #   X = np.reshape(x_tilde, [ns,nt])
        #   ax.imshow(X)
        #   plt.pause(0.01)

        X = np.reshape(x_tilde, [ns,nt])
        ax.imshow(X)
        plt.pause(0.01)

        if err_flag != 0:
            err = np.linalg.norm(H@x-b, ord=1)
            err_list.append(err)

        if time_flag != 0:
            t = toc()
            time_list.append(t)

        if value_flag != 0:
            value = c.T@x
            value_list.append(value)

    return x_tilde, err_list, time_list, value_list