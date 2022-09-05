# -*- coding: utf-8 -*-
"""
Various useful functions
"""

# Author: T.Tabuchi
# Date  : 2022/9/3
# References: https://github.com/PythonOT/POT.git

from functools import reduce
import time

import numpy as np

__time_tic_toc = time.time()


def tic():
    r""" Python implementation of Matlab tic() function """
    global __time_tic_toc
    __time_tic_toc = time.time()


def toc(message='Elapsed time : {} s'):
    r""" Python implementation of Matlab toc() function """
    t = time.time()
    print(message.format(t - __time_tic_toc))
    return t - __time_tic_toc


def compute_matrixH(ns,nt):
    Hr = np.repeat(np.eye(ns),nt).reshape(ns,ns*nt)
    Hc = np.tile(np.eye(nt),ns)
    H = np.vstack((Hr,Hc))
    return H

def creat_nonzerorow_index(H, ns, nt):
    row, col = np.nonzero(H)
    num_nonzero = row.shape[0] # (ns+nt)*ns
    ind = np.argsort(col)

    tmp = np.zeros(num_nonzero, dtype=np.int64)
    j = 0
    for i in ind:
        tmp[j] = row[i]
        j += 1
    return tmp.reshape(int(num_nonzero/2),2), col.reshape(ns+nt,nt)


def negatve_entropy(x):
    n = x.shape[0]
    s = 0
    for i in range(n):
        s = - x[i] * (np.log(x[i]) - 1)
    return s


def shuffle(ns,nt,maxIter):
    sampling_list = []
    M = int(maxIter/(ns+nt))
    if maxIter%(ns+nt) != 0:
        M += 1
    for i in range(M):
        tmp = np.arange(ns+nt)
        np.random.shuffle(tmp)
        sampling_list.append(tmp)
    return sampling_list