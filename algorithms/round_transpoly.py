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

    x = np.array([])
    for i in range(n):
        tmp = err_l * err_r[i]
        x = np.append(x, tmp)
    x = np.reshape(x, [n,n])
    A = A + x/sum(abs(err_r))

    return A

def round_transpoly2(T,r,l):
    A = T
    n = A.shape[0]
    r_A = np.sum(A,1)
    err1 = np.linalg.norm(r_A-r,ord=1)
    print("C_1 error: {}".format(err1))
    tmp = 0
    for i in range(n):
        if r[i] < r_A[i]:
            tmp += 1
        scaling = min(1,r[i]/r_A[i])
        A[i,:] = scaling * A[i,:]
    print(tmp)
    r_A = np.sum(A,1)
    err1 = np.linalg.norm(r_A-r,ord=1)
    print("C_1 error: {}".format(err1))

    l_A = np.sum(A,0)
    err2 = np.linalg.norm(l_A-l,ord=1)
    print("C_2 error: {}".format(err2))
    tmp = 0
    for j in range(n):
        if l[i] < l_A[i]:
            tmp += 1
        scaling = min(1,l[j]/l_A[j])
        A[:,j] = scaling * A[:,j]
    print(tmp)
    l_A = np.sum(A,0)
    err2 = np.linalg.norm(l_A-l,ord=1)
    print("C_2 error: {}".format(err2))
    
    r_A = np.sum(A,1)
    l_A = np.sum(A,0)
    err1 = np.linalg.norm(r_A-r,ord=1)
    print("C_1 error: {}".format(err1))
    err2 = np.linalg.norm(l_A-l,ord=1)
    print("C_2 error: {}".format(err2))
    err_r = r_A - r
    err_l = l_A - l

    x = np.array([])
    for i in range(n):
        tmp = err_l * err_r[i]
        x = np.append(x, tmp)
    x = np.reshape(x, [n,n])
    A = A + x/sum(abs(err_r))
    r_A = np.sum(A,1)
    err1 = np.linalg.norm(r_A-r,ord=1)
    print("C_1 error: {}".format(err1))
    l_A = np.sum(A,0)
    err2 = np.linalg.norm(l_A-l,ord=1)
    print("C_2 error: {}".format(err2))