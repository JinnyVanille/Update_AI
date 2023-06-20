# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd

def colnames(x, do_NULL=True, prefix="col"):
    if isinstance(x, pd.DataFrame) and do_NULL:
        return list(x.columns)
    else:
        dn = x.columns if isinstance(x, pd.DataFrame) else x.columns if x.ndim == 2 else None
        if dn is not None:
            return dn
        else:
            nc = x.shape[1]
            if do_NULL:
                return None
            elif nc > 0:
                return [prefix + str(i) for i in range(1, nc + 1)]
            else:
                return []