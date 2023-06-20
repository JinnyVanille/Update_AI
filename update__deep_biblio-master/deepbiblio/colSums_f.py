# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd
import numpy as np

def colSums(x, na_rm=False, dims=1):
    if isinstance(x, pd.DataFrame):
        x = x.values
    if not isinstance(x, np.ndarray) or len(x.shape) < 2:
        raise ValueError("'x' must be an array of at least two dimensions")
    if dims < 1 or dims > len(x.shape) - 1:
        raise ValueError("invalid 'dims'")
    n = np.prod(x.shape[id] for id in range(dims))
    dn = np.delete(x.shape, [id for id in range(dims)])
    if np.iscomplexobj(x):
        z = np.sum(x.real, axis=dims, dtype=np.complex128) + (0+1j) * np.sum(x.imag, axis=dims, dtype=np.complex128)
    else:
        z = np.sum(x, axis=dims, dtype=x.dtype)
    if len(dn) > 1:
        z.shape = tuple(dn)
        # Replace dimension names, if available
        if hasattr(x, 'dimnames') and dims < len(x.dimnames):
            dimnames = list(x.dimnames)
            dimnames[dims] = None
            z = np.array(z, dimnames)
    else:
        # Replace dimension names, if available
        if hasattr(x, 'dimnames') and dims < len(x.dimnames):
            names = x.dimnames[dims]
            if isinstance(names, list):
                names = np.array(names)
            z = np.array(z, names)
    return z