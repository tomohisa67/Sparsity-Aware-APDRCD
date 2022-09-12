from algorithms.apdrcd import apdrcd
from algorithms.sa_apdrcd import sa_apdrcd
from algorithms.pdrcd import pdrcd
from algorithms.round_transpoly import round_transpoly

import numpy as np
import matplotlib.pyplot as plt

# import sys
# import os

import ot
from ot.plot import plot1D_mat
from ot.lp import wasserstein_1d

from algorithms.utils import compute_matrixH


n = 100
ns = n
nt = n

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions for input
r = 0.5*ot.datasets.get_1D_gauss(n, m=70, sigma=9)+0.5*ot.datasets.get_1D_gauss(n, m=35, sigma=9)  # m= mean, s= std
l = 0.4*ot.datasets.get_1D_gauss(n, m=60, sigma=8)+0.6*ot.datasets.get_1D_gauss(n, m=40, sigma=6)

# loss matrix + normalization
scale = 1000
C = ot.utils.dist0(n) * scale
C /= C.max()

T = ot.sinkhorn(r, l, C, 1)

plot1D_mat(r,l,C)

reg = [1,5,9]
maxIter = 4000

x = np.array([])
err = np.array([])
tm = np.array([])
ot_loss = np.array([])

for eta in reg:
    xt, errt, tmt, ot_losst = apdrcd(r, l, C, eta, maxIter, err_flag=1, time_flag=1, value_flag=1)
    x = np.append(x, xt)
    err = np.append(err, errt)
    tm = np.append(tm, tmt)
    ot_loss = np.append(ot_loss, ot_losst)

L = len(reg)
x = np.reshape(x, [L,ns*nt])
err = np.reshape(err, [L,maxIter])
tm = np.reshape(tm, [L,maxIter])
ot_loss = np.reshape(ot_loss, [L,maxIter])

ii = np.arange(maxIter)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title("error")
for i in range(L):
    eta = reg[i]
    ax1.plot(ii, err[i], label="$\eta$ = {}".format(eta))
ax1.legend()
ax1.set_xlabel("iteration")
ax1.set_ylabel("$||Ax-b||_1$")

ax2.set_title("time")
for i in range(L):
    eta = reg[i]
    ax2.plot(ii, tm[i], label="$\eta$ = {}".format(eta))
ax2.legend()
ax2.set_xlabel("iteration")
ax2.set_ylabel("time (s)")

ax3.set_title("OT value")
for i in range(L):
    eta = reg[i]
    ax3.plot(ii, ot_loss[i], label="$\eta$ = {}".format(eta))
# tmp = ot_gt[0]
# ax3.plot(ii, ot_gt, label="ground_truth {:.2g}".format(tmp))
ax3.legend()
ax3.set_xlabel("iteration")
ax3.set_ylabel("ot_value")
plt.show()

fig1 = plt.figure()

ax1 = fig1.add_subplot(2, 4, 1)
ax2 = fig1.add_subplot(2, 4, 2)
ax3 = fig1.add_subplot(2, 4, 3)
ax4 = fig1.add_subplot(2, 4, 4)

ax5 = fig1.add_subplot(2, 4, 5)
ax6 = fig1.add_subplot(2, 4, 6)
ax7 = fig1.add_subplot(2, 4, 7)
ax8 = fig1.add_subplot(2, 4, 8)

ax1.set_title("Sinkhorn")
ax1.imshow(T)
ax5.set_title("Sinkhorn (round)")

X = np.reshape(x[0], [ns,nt])
X_round = round_transpoly(X,r,l)
ax2.imshow(X)
ax2.set_title("X : $\eta$ = {}".format(reg[0]))
ax6.imshow(X_round)
ax6.set_title("X_round: $\eta$ = {}".format(reg[0]))

X = np.reshape(x[0], [ns,nt])
X_round = round_transpoly(X,r,l)
ax3.imshow(X)
ax3.set_title("X : $\eta$ = {}".format(reg[1]))
ax7.imshow(X_round)
ax7.set_title("X_round: $\eta$ = {}".format(reg[1]))

X = np.reshape(x[0], [ns,nt])
X_round = round_transpoly(X,r,l)
ax4.imshow(X)
ax4.set_title("X : $\eta$ = {}".format(reg[2]))
ax8.imshow(X_round)
ax8.set_title("X_round: $\eta$ = {}".format(reg[2]))

plt.show()