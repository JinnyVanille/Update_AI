# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import panda as pd
import re
import numpy as np
from scipy.sparse import csr_matrix

def biblioNetwork(M, analysis="coupling", network="authors", n=None, sep=";", short=False, shortlabel=True, remove_terms=None, synonyms=None):
    cocMatrix = lambda x, Field, type, n, sep, short: csr_matrix((x[Field], (x["row"], x["col"])), shape=(n,n))

    NetMatrix = None
    if analysis == "coupling":
        if network == "authors":
            WA = cocMatrix(M, Field="AU", type="sparse", n=n, sep=sep, short=short)
            WCR = cocMatrix(M, Field="CR", type="sparse", n=n, sep=sep, short=short)
            CRA = WCR.T.dot(WA)
            NetMatrix = CRA.T.dot(CRA)
        elif network == "references":
            WCR = cocMatrix(M, Field="CR", type="sparse", n=n, sep=sep, short=short).T
            NetMatrix = WCR.dot(WCR.T)
        elif network == "sources":
            WSO = cocMatrix(M, Field="SO", type="sparse", n=n, sep=sep, short=short)
            WCR = cocMatrix(M, Field="CR", type="sparse", n=n, sep=sep, short=short)
            CRSO = WCR.T.dot(WSO)
            NetMatrix = CRSO.T.dot(CRSO)
        elif network == "countries":
            WCO = cocMatrix(M, Field="AU_CO", type="sparse", n=n, sep=sep, short=short)
            WCR = cocMatrix(M, Field="CR", type="sparse", n=n, sep=sep, short=short)
            CRCO = WCR.T.dot(WCO)
            NetMatrix = CRCO.T.dot(CRCO)

    if analysis == "co-occurrences":
        if network == "authors":
            WA = cocMatrix(M, Field="AU", type="sparse", n=n, sep=sep, short=short)
        elif network == "keywords":
            WA = cocMatrix(M, Field="ID", type="sparse", n=n, sep=sep, short=short, remove_terms=remove_terms,
                           synonyms=synonyms)
        elif network == "author_keywords":
            WA = cocMatrix(M, Field="DE", type="sparse", n=n, sep=sep, short=short, remove_terms=remove_terms,
                           synonyms=synonyms)
        elif network == "titles":
            WA = cocMatrix(M, Field="TI_TM", type="sparse", n=n, sep=sep, short=short, remove_terms=remove_terms,
                           synonyms=synonyms)
        elif network == "abstracts":
            WA = cocMatrix(M, Field="AB_TM", type="sparse", n=n, sep=sep, short=short, remove_terms=remove_terms,
                           synonyms=synonyms)
        elif network == "sources":
            WA = cocMatrix(M, Field="SO", type="sparse", n=n, sep=sep, short=short)
        NetMatrix = WA.T.dot(WA)

    if analysis == "co-citation":
        if network == "authors":
            WA = cocMatrix(M, Field="CR_AU", type="sparse", n=n, sep=sep, short=short)
        elif network == "references":
            WA = cocMatrix(M, Field="CR", type="sparse", n=n, sep=sep, short=short)
        elif network == "sources":
            WA = cocMatrix(M, Field="CR_SO", type="sparse", n=n, sep=sep, short=short)
        NetMatrix = WA.T.dot(WA)

    if analysis == "collaboration":
        if network == "authors":
            WA = cocMatrix(M, Field="AU", type="sparse", n=n, sep=sep, short=short)
        elif network == "universities":
            WA = cocMatrix(M, Field="AU_UN", type="sparse", n=n, sep=sep, short=short)
        elif network == "countries":
            WA = cocMatrix(M, Field="AU_CO", type="sparse", n=n, sep=sep, short=short)
        NetMatrix = WA.T.dot(WA)

    # delete empty vertices
    row_mask = np.array([len(str(i)) != 0 for i in NetMatrix.index])
    col_mask = np.array([len(str(i)) != 0 for i in NetMatrix.columns])
    NetMatrix = NetMatrix.loc[row_mask, col_mask]

    # short label for scopus references
    if network == "references" and M.DB[0] == "SCOPUS":
        ind = np.where(np.char.isalpha(NetMatrix.columns.str.slice(0, 1)))[0]
        NetMatrix = NetMatrix.iloc[ind, ind]

    if network == "references" and shortlabel:
        LABEL = labelShort(NetMatrix, db=M.DB[0].lower())
        LABEL = removeDuplicatedlabels(LABEL)
        NetMatrix.columns = NetMatrix.index = LABEL
    return NetMatrix

def labelShort(NET, db="isi"):
    LABEL = NET.columns
    YEAR = [re.findall('\d{4}', l)[-1] if re.findall('\d{4}', l) else '' for l in LABEL]
    if db == "isi":
        AU = [l.split()[0] + ' ' + l.split()[1] for l in LABEL]
        LABEL = [AU[i] + ' ' + YEAR[i] for i in range(len(LABEL))]
    elif db == "scopus":
        AU = [l.split('. ')[0] for l in LABEL]
        LABEL = [AU[i] + '. ' + YEAR[i] for i in range(len(LABEL))]
    return LABEL

def removeDuplicatedlabels(LABEL):
    # assign a unique name to each label
    tab = pd.Series(LABEL).value_counts()
    dup = tab[tab > 1].index
    for i in range(len(dup)):
        ind = np.where(LABEL == dup[i])[0]
        if len(ind) > 0:
            new_labels = [f"{label}-{j+1}" for j, label in enumerate(LABEL[ind])]
            LABEL[ind] = new_labels
    return LABEL