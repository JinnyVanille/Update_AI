# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

import pandas as pd
import numpy as np
from datetime import datetime

def Hindex(M, field="author", elements=None, sep=";", years=np.inf):
    M["TC"] = pd.to_numeric(M["TC"], errors="coerce")
    M["PY"] = pd.to_numeric(M["PY"], errors="coerce")
    M = M.dropna(subset=["TC", "PY"])

    Today = datetime.now().year
    past = Today - years
    M = M[M["PY"] >= past]

    if field == "author":
        AU = M["AU"]
        AU = AU.str.replace(",", " ")
        listAU = AU.str.split(pat=sep, expand=True)
        l = listAU.count(axis=1)
        index = np.repeat(M.index, l)
        df = M.loc[index, :]
        df["AUs"] = listAU.values.flatten()
    elif field == "source":
        df = M.copy()
        df["AUs"] = M["SO"]

    def h_calc(x):
        h = np.sum(x >= np.sort(x)[::-1].cumsum()) + 1
        return h

    def g_calc(x):
        g = np.sum(x >= np.sort(x)[::-1].cumsum() / np.arange(1, len(x) + 1)[::-1]) + 1
        return g

    H = df.groupby("AUs").agg({
        "PY": "min",
        "TC": ["sum", h_calc, g_calc],
        "AUs": "count"
    })
    H.columns = ["PY_start", "TC", "h_index", "g_index", "NP"]
    H["m_index"] = H["h_index"] / (Today - H["PY_start"] + 1)
    H = H.reset_index()

    if elements is not None:
        H = H[H["AUs"].isin(elements)]
        df = df[df["AUs"].isin(elements)]

    CitationList = df.groupby("AUs")[["AU", "SO", "PY", "TC", "DI"]].apply(lambda x: x.to_dict("records")).to_dict()

    results = {"H": H, "CitationList": CitationList}

    return results