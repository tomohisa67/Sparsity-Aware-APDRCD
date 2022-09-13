# -*- coding: utf-8 -*-
"""
Load MNIST dataset
"""
# Author: T.Tabuchi
# Date  : 2022/9/12

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import fetch_openml


def load_mnist784():
    save_file = 'dataset/mnist/mnist.pkl'
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    r = dataset['train_img'][0].astype(np.float64)
    r = r / sum(r)
    l = dataset['train_img'][1].astype(np.float64)
    l = l / sum(l)
    return r,l


def load_mnist784_2():
    digits = fetch_openml(name='mnist_784', version=1)
    data = digits.data.to_numpy()
    return data


def load_mnist784_random():
    digits = fetch_openml(name='mnist_784', version=1)
    data = digits.data.to_numpy()
    N = data.shape[0]
    tmp = np.arange(N)
    # np.random.seed(111)
    np.random.shuffle(tmp)
    i,j = tmp[0:2]
    return data[i], data[j]


def creat_mnist_costmat():
    for i in range(28):
        for j in range(28):
            C[i][j] = np.abs(i-j)
    return C