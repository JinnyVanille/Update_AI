# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import numpy as np

def rowSums(x, na_rm=False):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim < 2:
        raise ValueError("'x' must be an array of at least two dimensions")
    axis = 1
    if na_rm:
        return np.nansum(x, axis=axis)
    else:
        return np.sum(x, axis=axis)