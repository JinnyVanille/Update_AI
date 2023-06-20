# author:  Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import re
from igraph import Graph
from collections import defaultdict
import pandas as pd
import numpy as np
from deepbiblio.metaTagExtraction_f import metaTagExtraction

def histNetwork(M, min_citations=1, sep=';', network=True, verbose=True):
    db = M['DB'][0]

    if 'DI' not in M:
        M['DI'] = ''
    else:
        M['DI'].fillna('', inplace=True)

    if 'CR' not in M:
        print('\nYour collection does not contain Cited References metadata (Field CR is missing)\n')
        return None

    M['TC'].fillna(0, inplace=True)

    if db == 'ISI':
        db = 'WOS'

    if db == 'WOS':
        results = wos(M=M, min_citations=min_citations, sep=sep, network=network, verbose=verbose)
    elif db == 'SCOPUS':
        results = scopus(M=M, min_citations=min_citations, sep=sep, network=network, verbose=verbose)
    else:
        print('\nDatabase not compatible with direct citation analysis\n')
        return None

    return results

def wos(M, min_citations, sep, network, verbose):
    if verbose:
        print("\nWOS DB:\nSearching local citations (LCS) by reference items (SR) and DOIs...\n")

    if 'SR_FULL' not in M.columns:
        M = metaTagExtraction(M, Field="SR")

    M = M.sort_values('PY')
    M['Paper'] = np.arange(1, len(M) + 1)
    M_orig = M.copy()
    M['nLABEL'] = np.arange(1, len(M) + 1)

    # Reference list and citing papers
    CR = M['CR'].str.split(sep, expand=True)
    CR = CR.stack().reset_index(level=1, drop=True).reset_index(name='ref')
    CR['paper'] = CR['level_0'] + 1
    CR['DI'] = CR['ref'].str.extract('DOI\s*(.*)', expand=False).str.strip()
    CR['DI'].fillna('', inplace=True)
    CR['AU'] = CR['ref'].str.split(',', n=1, expand=True)[0].str.replace('.', ' ').str.strip()
    CR['PY'] = CR['ref'].str.split(',', n=2, expand=True)[1].str.strip()
    CR['SO'] = CR['ref'].str.split(',', n=2, expand=True)[2].str.strip()
    CR['SR'] = CR['AU'] + ', ' + CR['PY'] + ', ' + CR['SO']

    if verbose:
        print("\nWOS DB:\nSearching local citations (LCS) by reference items (SR) and DOIs...\n")

    if "SR_FULL" not in M.columns:
        M = metaTagExtraction(M, Field="SR")

    M = M.sort_values("PY")
    M["Paper"] = range(1, M.shape[0] + 1)
    M_orig = M.copy()
    M["nLABEL"] = range(1, M.shape[0] + 1)

    # Reference list and citing papers
    CR = M["CR"].str.split(sep)
    CR = pd.concat([pd.DataFrame({"ref": r, "paper": i}) for i, r in enumerate(CR)], ignore_index=True)
    CR["DI"] = CR["ref"].str.split("DOI").str[1].str.strip()
    CR["DI"].loc[(CR["DI"].isna()) | (CR["DI"] == "NA")] = ""
    CR["AU"] = CR["ref"].str.split(",", n=1, expand=True)[0].str.replace(".", " ").str.replace("  ", " ").str.strip()
    CR["PY"] = CR["ref"].str.split(",", n=2, expand=True)[1].str.strip()
    CR["SO"] = CR["ref"].str.split(",", n=2, expand=True)[2].str.strip()
    CR["SR"] = CR["AU"] + ", " + CR["PY"] + ", " + CR["SO"]

    if verbose:
        print("\nAnalyzing", CR.shape[0], "reference items...\n")

    # Local cited documents by DOI and reference item
    M["LABEL"] = M["SR_FULL"] + "DOI" + M["DI"].str.upper()
    CR["LABEL"] = CR["SR"] + "DOI" + CR["DI"]

    # By reference
    L = pd.merge(M, CR, on="LABEL", how="left")
    L = L.loc[~L["paper"].isna()]
    L["CITING"] = L["LABEL"].loc[L["paper"].astype(int)].values
    L["nCITING"] = L["nLABEL"].loc[L["paper"].astype(int)].values
    L["CIT_PY"] = L["PY"].loc[L["paper"].astype(int)].values

    LCS = L.groupby("nLABEL").agg({"LABEL": "first", "nLABEL": "size"}).reset_index()[["LABEL", "nLABEL"]]
    LCS.columns = ["LABEL", "n"]

    M["LCS"] = 0
    M.loc[LCS["nLABEL"] - 1, "LCS"] = LCS["n"].values
    M_orig["LCS"] = M["LCS"]

    histData = M[["LABEL", "TI", "DE", "ID", "DI", "PY", "LCS", "TC"]]
    histData.columns = ["Paper", "Title", "Author_Keywords", "KeywordsPlus", "DOI", "Year", "LCS", "GCS"]

    if network:
        if verbose:
            print("\nCreating the citing data frame...")

        # Citing data frame
        CITING = (L.groupby('CITING')
                  .agg({'LABEL': lambda x: ';'.join(x),
                        'CIT_PY': 'first',
                        'paper': 'first'})
                  .reset_index()
                  .sort_values('CIT_PY')
                  .reset_index(drop=True))
        CITING.columns = ['CITING', 'LCR', 'PY', 'Paper']

        # Add LCR column to M_orig
        M_orig['LCR'] = np.nan
        M_orig.loc[CITING['Paper'], 'LCR'] = CITING['LCR']

        # Assign an unique name to each document
        i = 0
        while True:
            ind = np.where(M.duplicated('LABEL'))[0]
            if len(ind) > 0:
                i += 1
                M.loc[ind, 'LABEL'] = M.loc[ind, 'LABEL'] + '-' + chr(ord('a') + i - 1)
            else:
                break

        M.index = M['LABEL']
        LABEL = M.index

        # NetMatrix
        WLCR = pd.crosstab(index=M['LABEL'], columns=M['LCR'].str.split(';').explode()).to_numpy()
        missing_LABEL = np.setdiff1d(LABEL, np.array(WLCR.columns))
        WLCR = np.hstack([WLCR, np.zeros((len(WLCR), len(missing_LABEL)))])
        WLCR = pd.DataFrame(WLCR, columns=np.concatenate([WLCR.columns, missing_LABEL]), index=LABEL)
        WLCR = WLCR.loc[LABEL, missing_LABEL]
        WLCR = WLCR.to_numpy()

    else:
        WLCR = None

    if verbose:
        print("\nFound", sum(M['LCS'] > 0), "documents with no empty Local Citations (LCS)")

    histData = M.loc[:, ['LABEL', 'TI', 'DE', 'ID', 'DI', 'PY', 'LCS', 'TC']]
    histData.columns = ['Paper', 'Title', 'Author_Keywords', 'KeywordsPlus', 'DOI', 'Year', 'LCS', 'GCS']

    results = {
        'NetMatrix': WLCR,
        'histData': histData,
        'M': M_orig,
        'LCS': M['LCS']
    }

    return results

def scopus(M, min_citations, sep, network, verbose):
    if verbose:
        print("\nSCOPUS DB: Searching local citations (LCS) by document titles (TI) and DOIs...\n")

    M['nCITING'] = np.arange(1, len(M) + 1)
    papers = M.loc[M['TC'] >= min_citations, 'nCITING'].values

    TIpost = [re.sub(r'[^\w\s]', '', M.loc[i, 'TI']) + ' ' + str(M.loc[i, 'PY']) + ' ' for i in papers]
    CR = re.sub(r'[^\w\s]', '', ' '.join(M['CR'].values))
    n = np.array([len(i) for i in CR])
    n[np.isnan(n)] = 2
    n += 1
    n_cum = np.concatenate(([1], np.cumsum(n[:-1])))
    CR = ' '.join(CR)

    L = [np.array(x) for x in re.findall(f'(?=({"|".join(TIpost)})).', CR)]
    L = [x for x in L if x.size > 0]
    LCS = np.array([len(x) for x in L]) / 2

    M['LCS'] = 0
    M.loc[papers - 1, 'LCS'] = LCS

    if verbose:
        print(f"\nFound {len(M[M['LCS'] > 0])} documents with no empty Local Citations (LCS)\n")

    if network:
        df = defaultdict(list)
        for i, l in enumerate(L):
            df['ref'].extend(l.tolist())
            df['paper'].extend([papers[i]] * len(l))
        df = pd.DataFrame(df)

        A = np.subtract.outer(df['ref'], n_cum)
        A[A < 0] = np.nan
        df['CITINGn'] = np.nanargmin(A, axis=1)
        df['CITING'] = M['SR'][df['CITINGn']].values
        df['CITED'] = M['SR'][df['paper']].values

        NetMatrix = Graph.TupleList(df[['CITING', 'CITED']].itertuples(index=False), directed=True).get_adjacency().data
    else:
        NetMatrix = None

    histData = M.loc[:, ['SR_FULL', 'TI', 'DE', 'ID', 'DI', 'PY', 'LCS', 'TC']]
    histData.columns = ['Paper', 'Title', 'Author_Keywords', 'KeywordsPlus', 'DOI', 'Year', 'GCS']
    histData = histData.sort_values('Year').reset_index(drop=True)

    histData = (M
                .loc[:, ['SR_FULL', 'TI', 'DE', 'ID', 'DI', 'PY', 'LCS', 'TC']]
                .rename(columns={
        'SR_FULL': 'Paper',
        'TI': 'Title',
        'DE': 'Author_Keywords',
        'ID': 'KeywordsPlus',
        'DI': 'DOI',
        'PY': 'Year',
        'TC': 'GCS'
    })
                .sort_values('Year')
                )

    if network:
        papers = M.index[M['TC'] >= min_citations].tolist()
        TIpost = (M.loc[papers, 'TI']
                  .str.replace("[[:punct:]]", "", regex=True)
                  .str.cat(M.loc[papers, 'PY'].astype(str), sep=' ')
                  )

        CR = M['CR'].str.replace("[[:punct:]]", "", regex=True).fillna('').tolist()
        n = [len(x) + 1 for x in CR]
        n_cum = [1] + np.cumsum(n[:-1]).tolist()
        CR = ' '.join(CR)

        L = [(m.start(), m.end()) for m in re.finditer(TIpost, CR)]
        LCS = [len(x) / 2 for x in L]

        df = pd.concat([pd.DataFrame({'ref': x, 'paper': np.repeat(papers[i], len(x))})
                        for i, x in enumerate(L)])
        A = np.subtract.outer(df['ref'].values, n_cum)
        A[A < 0] = np.nan
        df['CITINGn'] = np.apply_along_axis(np.argmin, 1, A)
        df['CITING'] = M.loc[df['CITINGn'], 'SR'].values
        df['CITED'] = M.loc[df['paper'], 'SR'].values

        NetMatrix = Graph.TupleList(df.loc[:, ['CITING', 'CITED']].itertuples(index=False),
                                    directed=True).get_adjacency().data
    else:
        NetMatrix = None

    if verbose:
        print(f"\nFound {len(M.loc[M['LCS'] > 0])} documents with no empty Local Citations (LCS)\n")

    results = {
        'NetMatrix': NetMatrix,
        'histData': histData,
        'M': M,
        'LCS': M['LCS']
    }


    return results