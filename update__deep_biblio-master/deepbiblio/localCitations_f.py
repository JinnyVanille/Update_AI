# author:  Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
from deepbiblio.histNetwork_f import histNetwork


def localCitations(M, fast_search=False, sep=";", verbose=False):
    import pandas as pd
    from collections import Counter

    M["TC"].fillna(0, inplace=True)
    if fast_search:
        loccit = M["TC"].quantile(0.75, interpolation="nearest")
    else:
        loccit = 1

    H = histNetwork(M, min_citations=loccit, sep=sep, network=False, verbose=verbose)
    LCS = H["histData"]
    M = H["M"]
    del H
    AU = [a.split(sep) for a in M["AU"]]
    n = [len(a) for a in AU]

    df = pd.DataFrame({"AU": [a for b in AU for a in b], "LCS": LCS["LCS"].repeat(n)})
    AU = df.groupby("AU")["LCS"].sum().reset_index().rename(
        columns={"AU": "Author", "LCS": "LocalCitations"}).sort_values("LocalCitations", ascending=False)

    if "SR" in M.columns:
        LCS = M[["SR", "DI", "PY", "LCS", "TC"]].rename(
            columns={"SR": "Paper", "DI": "DOI", "PY": "Year", "LCS": "LCS", "TC": "GCS"}).sort_values("LCS",
                                                                                                       ascending=False)
    else:
        LCS = pd.DataFrame()

    CR = {"Authors": AU, "Papers": LCS, "M": M}
    return CR
