from algorithms.apdrcd import apdrcd
from algorithms.sa_apdrcd import sa_apdrcd
from algorithms.pdrcd import pdrcd

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

import ot
from ot.plot import plot1D_mat


n = 100
ns = n
nt = n

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions for input
r = 0.5*ot.datasets.get_1D_gauss(n, m=70, sigma=9)+0.5*ot.datasets.get_1D_gauss(n, m=35, sigma=9)  # m= mean, s= std
l = 0.4*ot.datasets.get_1D_gauss(n, m=60, sigma=8)+0.6*ot.datasets.get_1D_gauss(n, m=40, sigma=6)

# loss matrix + normalization
C = ot.utils.dist0(n)
C /= C.max()

T = ot.sinkhorn(r, l, C, 1)

plot1D_mat(r,l,C)

eta = 1
maxIter = 500

x1 = np.zeros(ns*nt)
x2 = np.zeros(ns*nt)
ii = np.arange(maxIter)
err1 = []
err2 = []
time1 = []
time2 = []
ot1 = []
ot2 = []

x1, err1, time1, ot1 = apdrcd(r, l, C, eta, maxIter, err_flag=1, time_flag=1, value_flag=1)
# x2, err2, time2, ot2 = sa_apdrcd(r, l, C, eta, maxIter, err_flag=1, time_flag=1, value_flag=1)
x3, err3, time3, ot3 = pdrcd(r, l, C, eta, maxIter, err_flag=1, time_flag=1, value_flag=1)

fig, ax = plt.subplots(facecolor="w")
ax.plot(ii, err1, label="original")
# ax.plot(ii, err2, label="sparsity-aware")
# ax.plot(ii, err3, label="pdrcd")
ax.legend()
ax.set_xlabel("iteration")
ax.set_ylabel("$||Ax-b||_1$")
plt.show()

fig, ax = plt.subplots(facecolor="w")
ax.plot(ii, time1, label="original")
# ax.plot(ii, time2, label="sparsity-aware")
# ax.plot(ii, time3, label="pdrcd")
ax.legend()
ax.set_xlabel("iteration")
ax.set_ylabel("time (s)")
plt.show()

fig, ax = plt.subplots(facecolor="w")
ax.plot(ii, ot1, label="original")
# ax.plot(ii, ot2, label="sparsity-aware")
# ax.plot(ii, ot3, label="pdrcd")
ax.legend()
ax.set_xlabel("iteration")
ax.set_ylabel("ot_value")
plt.show()

x1 = np.ones(ns*nt)
X = np.reshape(x1, [ns,nt])

fig = plt.figure(figsize=(8,6))
plt.imshow(X)
plt.title("X")
plt.show()

print(x1)