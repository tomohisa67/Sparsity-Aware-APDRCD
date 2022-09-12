# -*- coding: utf-8 -*-
"""
round transport polytope
"""
# Author: T.Tabuchi
# Date  : 2022/9/12
# References: https://github.com/JasonAltschuler/OptimalTransportNIPS17/blob/master/algorithms/round_transpoly.m

import numpy as np

def round_transpoly(T,r,l):
    A = T
    n = A.shape[0]
    r_A = np.sum(A,1)
    for i in range(n):
        scaling = min(1,r[i]/r_A[i])
        A[i,:] = scaling * A[i,:]

    l_A = np.sum(A,0)
    for j in range(n):
        scaling = min(1,l[j]/l_A[j])
        A[:,j] = scaling * A[:,j]
    
    r_A = np.sum(A,1)
    l_A = np.sum(A,0)
    err_r = r_A - r
    err_l = l_A - l

    A = A + err_r*err_l/sum(abs(err_r))

    return A