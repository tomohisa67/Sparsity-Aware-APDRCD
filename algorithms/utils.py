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
