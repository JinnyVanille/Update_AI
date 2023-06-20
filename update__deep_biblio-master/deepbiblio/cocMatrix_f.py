# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import re
import pandas as pd
import numpy as np
from collections import Counter
import scipy.sparse as sp

from deepbiblio.trimES_f import trimES


def cocMatrix(M, Field="AU", type="sparse", n=None, sep=";", binary=True, short=False, remove_terms=None,
              synonyms=None):
    size = M.shape
    if "LABEL" not in M.columns:
        M.index = M["SR"]
    RowNames = M.index

    ### REMOVE TERMS AND MERGE SYNONYMS
    if Field in ["ID", "DE", "TI", "TI_TM", "AB", "AB_TM"]:
        # Create df with all terms
        Fi = M[Field].str.split(sep)
        TERMS = pd.DataFrame(
            {"item": [t.strip() for sublist in Fi for t in sublist], "SR": M["SR"].repeat(Fi.apply(len))})

        # Merge synonyms in the vector synonyms
        if synonyms and isinstance(synonyms, str):
            s = [t.upper().split(";") for t in synonyms.split()]
            snew = [t[0].strip() for t in s]
            sold = [list(map(str.strip, t[1:])) for t in s]
            for i in range(len(s)):
                TERMS.loc[TERMS["item"].isin(sold[i]), "item"] = snew[i]

        TERMS = TERMS[~TERMS["item"].str.upper().isin(remove_terms)]
        TERMS["item"] = TERMS["item"].str.strip()

        TERMS = TERMS.groupby("SR")["item"].apply(lambda x: ";".join(x)).reset_index()
        M = pd.merge(M, TERMS, on="SR", how="left")
        M[Field] = M["item"]
        M.index = RowNames

    if Field == "CR":
        M["CR"] = M["CR"].str.replace("DOI;", "DOI ")

    if Field in M.columns:
        Fi = M[Field].str.split(sep)
    else:
        return print(f"Field {Field} is not a column name of input data frame")

    Fi = Fi.apply(lambda x: [t.lstrip() for t in x])
    if Field == "CR":
        Fi = Fi.apply(lambda x: [t for t in x if len(t) > 10])

    ## Scelta dell'informazione contenuta in CR da utilizzare (Reference, Autore, Affiliation, ecc.)

    # vector of unique units
    allField = [t for sublist in Fi for t in sublist if pd.notna(t)]

    # normalize reference names
    if Field == "CR":
        ind = [i for i, x in enumerate(allField) if x[0] != "("]
        S = allField.copy()
        for i in ind:
            S[i] = re.sub("\\).*", ")", allField[i])
        S = [re.sub(",", " ", x) for x in S]
        S = [re.sub(";", " ", x) for x in S]
        S = reduce_refs(S)
        allField = trimES(S)
        Fi = [reduce_refs([re.sub("\\).*", ")", re.sub(",", " ", re.sub(";", " ", x)))
                          for x in l if len(x) > 0]) for l in Fi]

    else:
        S = [re.sub(",", ";", x) for x in allField]
        S = [re.sub("\\;", ",", x) for x in S]
        S = [re.sub("\\;", ",", x) for x in S]
        S = [re.sub("\\;.*", "", x) for x in S]
        allField = (S).strip()
        Fi = [([re.sub("\\,", ";", re.sub("\\;", ",", re.sub("\\;.*", "", x)))
                      for x in l if len(x) > 0]).strip() for l in Fi]

    tabField = dict(sorted(Counter(allField).items(), key=lambda item: item[1], reverse=True))
    uniqueField = list(tabField.keys())

    # select n items
    if n is not None:
        uniqueField = uniqueField[:n]
    elif short:
        uniqueField = [k for k, v in tabField.items() if v > 1]

    if len(uniqueField) < 1:
        print("Matrix is empty!!")
        return None

    if type == "matrix" or not binary:
        # Initialization of WA matrix
        WF = np.zeros((M.shape[0], len(uniqueField)), dtype=int)
    elif type == "sparse":
        WF = sp.sparse.lil_matrix((M.shape[0], len(uniqueField)), dtype=int)
    else:
        print("error in type argument")
        return None

    for i in range(M.shape[0]):
        if len(Fi[i]) > 0 and Fi[i][0] is not None:
            if binary:
                ind = [uniqueField.index(x) for x in Fi[i] if x in uniqueField]
                if len(ind) > 0:
                    WF[i, ind] = 1
            else:
                tab = Counter(Fi[i])
                name = [k for k in tab.keys() if k in uniqueField and len(k) > 0]
                if len(name) > 0:
                    ind = [uniqueField.index(k) for k in name]
                    WF[i, ind] = [tab[k] for k in name]

    if type == "sparse" and not binary:
        WF = sp.sparse.csr_matrix(WF)

    WF = WF[:, [i for i, x in enumerate(uniqueField) if x != "NA"]]

    return WF

def reduce_refs(A):
    ind = [m.end() for m in re.finditer('V[0-9]', A)]
    A[ind > -1] = [s[:i-1] for i, s in zip(ind, A[ind > -1])]
    ind = [m.end() for m in re.finditer('DOI ', A)]
    A[ind > -1] = [s[:i-1] for i, s in zip(ind, A[ind > -1])]
    return A