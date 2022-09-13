from algorithms.apdrcd import apdrcd
from algorithms.sa_apdrcd import sa_apdrcd
from algorithms.pdrcd import pdrcd
from algorithms.round_transpoly import round_transpoly
from dataset.load_data import load_mnist784

import numpy as np
import matplotlib.pyplot as plt

import ot
from ot.plot import plot1D_mat
from ot.lp import wasserstein_1d

# from algorithms.utils import compute_marginal_err

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

### MNIST ###
# r,l = load_mnist784()
# C = np.zeros((ns,nt))
# for i in range(28):
#     for j in range(28):
#         C[i][j] = np.abs(i-j)

plot1D_mat(r,l,C)

reg = 0.01
maxIter = 4000

T = ot.sinkhorn(r, l, C, 1)

# x1 = np.zeros(ns*nt)
# x2 = np.zeros(ns*nt)
# x3 = np.zeros(ns*nt)
ii = np.arange(maxIter)
# err1 = []
# err2 = []
# err3 = []
# time1 = []
# time2 = []
# time3 = []
# ot1 = []
# ot2 = []
# ot3 = []

x1, err1, time1, ot1 = apdrcd(r, l, C, reg, maxIter, err_flag=1, time_flag=1, value_flag=1)
x2, err2, time2, ot2 = sa_apdrcd(r, l, C, reg, maxIter, err_flag=1, time_flag=1, value_flag=1)
x3, err3, time3, ot3 = pdrcd(r, l, C, reg, maxIter, err_flag=1, time_flag=1, value_flag=1)
ot_loss = wasserstein_1d(r,l) ###################################################################### LP solver に変更する
ot_loss = np.tile(ot_loss, maxIter)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title("error")
ax1.plot(ii, err1, label="original")
ax1.plot(ii, err2, label="sparsity-aware")
ax1.plot(ii, err3, label="pdrcd")
ax1.legend()
ax1.set_xlabel("iteration")
ax1.set_ylabel("$||Ax-b||_1$")

ax2.set_title("Time")
ax2.plot(ii, time1, label="original")
ax2.plot(ii, time2, label="sparsity-aware")
ax2.plot(ii, time3, label="pdrcd")
ax2.legend()
ax2.set_xlabel("iteration")
ax2.set_ylabel("time (s)")

ax3.set_title("OT value")
ax3.plot(ii, ot1, label="original")
ax3.plot(ii, ot2, label="sparsity-aware")
ax3.plot(ii, ot3, label="pdrcd")
ax3.plot(ii, ot_loss, label="ground_truth {:.2g}".format(ot_loss[0]))
ax3.legend()
ax3.set_xlabel("iteration")
ax3.set_ylabel("ot_value")
plt.show()

X = np.reshape(x1, [ns,nt])
X2 = np.reshape(x2, [ns,nt])
X3 = np.reshape(x3, [ns,nt])

X_round = round_transpoly(X,r,l)
X2_round = round_transpoly(X2,r,l)
X3_round = round_transpoly(X3,r,l)

# tt = T.flatten()
# H = compute_matrixH(ns,nt)
# b = np.concatenate((r,l))
# errs = np.linalg.norm(H@tt-b,ord=1)
# print(errs)
# erra = np.linalg.norm(H@x1-b,ord=1)
# print(erra)

# from algorithms.bregman import sinkhorn

# Tm = sinkhorn(r,l,C,1)
# fig = plt.figure(figsize=(8,6))
# plt.imshow(Tm)
# plt.title("Tm")
# plt.show()

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
T_round = round_transpoly(T,r,l)
ax5.imshow(T_round)
ax5.set_title("Sinkhorn (round)")

ax2.set_title("original")
ax2.imshow(X)
ax6.set_title("original (round)")
ax6.imshow(X_round)

ax3.set_title("sparsity-aware")
ax3.imshow(X2)
ax7.set_title("sparsity-aware (round)")
ax7.imshow(X2_round)

ax4.set_title("pdrcd")
ax4.imshow(X3)
ax8.set_title("pdrcd (round)")
ax8.imshow(X3_round)

plt.show()