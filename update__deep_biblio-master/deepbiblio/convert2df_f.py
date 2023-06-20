# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import re
import chardet
import numpy as np
import pandas as pd
from deepbiblio.bib2df_f import bib2df
from deepbiblio.cochrane2df_f import cochrane2df
from deepbiblio.csvLens2df_f import csvLens2df
from deepbiblio.csvScopus2df_f import csvScopus2df
from deepbiblio.dimensions2df_f import dimensions2df
from deepbiblio.isi2df_f import isi2df
from deepbiblio.metaTagExtraction_f import metaTagExtraction
from deepbiblio.pubmed2df_f import pubmed2df
from deepbiblio.trimES_f import trimES


def convert2df(file, dbsource="wos", format="plaintext"):
    allowed_formats = ['api', 'bibtex', 'csv', 'endnote', 'excel', 'plaintext', 'pubmed']
    allowed_db = ['cochrane', 'dimensions', 'generic', 'isi', 'pubmed', 'scopus', 'wos', 'lens']
    M = None
    print(f"\nConverting your {dbsource} collection into a bibliographic dataframe\n")

    if dbsource not in allowed_db:
        print(f"\n 'dbsource' argument is not properly specified")
        print(f"\n 'dbsource' argument has to be a character string matching one among: {allowed_db}")
        return

    if format not in allowed_formats:
        print(f"\n 'format' argument is not properly specified")
        print(f"\n 'format' argument has to be a character string matching one among: {allowed_formats}")
        return

    if format not in ["api", "plaintext", "bibtex", "csv", "excel", "endnote"]:
        with open(file, 'rb') as f:
            result = chardet.detect(f.read())
            file_encoding = result['encoding']
        with open(file, encoding=file_encoding) as f:
            D = f.read()
            D = D.encode('ascii', 'ignore').decode()
    else:
        with open(file, encoding='utf-8') as f:
            D = f.read()

    if dbsource == "wos":
        dbsource = "isi"
    if format == "endnote":
        format = "plaintext"
    if format == "lens":
        format = "csv"

    if dbsource == "isi":
        if format == "bibtex":
            M = bib2df(D, dbsource="isi")
        elif format == "plaintext":
            M = isi2df(D)

    elif dbsource == "scopus":
        if format == "bibtex":
            M = bib2df(D, dbsource="scopus")
        elif format == "csv":
            M = csvScopus2df(file)

    # elif dbsource == "generic":
    #     M = bib2df(D, dbsource="generic")
    #
    # elif dbsource == "lens":
    #     M = csvLens2df(file)

    # elif dbsource == "pubmed":
    #     if format == "api":
    #         M = pmApi2df(file)
    #         M['DB'] = 'PUBMED'
    #     else:
    #         M = pubmed2df(D)
    # elif dbsource == "pubmed":
    #     if format != "api":
    #         M = pubmed2df(D)
    #
    # elif dbsource == "cochrane":
    #     M = cochrane2df(D)

    # elif dbsource == "dimensions":
    #     if format == "api":
    #         M = dsApi2df(file)
    #         M['DB'] = 'DIMENSIONS'
    #     else:
    #         M = dimensions2df(file, format=format)

    # elif dbsource == "dimensions":
    #     if format != "api":
    #         M = dimensions2df(file, format=format)
    #
    # M = pd.read_csv(file)

    if "PY" in M.columns:
        M["PY"] = pd.to_numeric(M["PY"], errors="coerce")
    else:
        M["PY"] = np.nan

    if "TC" in M.columns:
        M["TC"] = pd.to_numeric(M["TC"], errors="coerce")
        M["TC"].fillna(0, inplace=True)
    else:
        M["TC"] = 0

    if "CR" not in M.columns:
        M["CR"] = "none"
    else:
        M["CR"] = M["CR"].str.replace("\\[,||\\[||\\]|| \\.\\. || \\. ", "")
        for x in range(len(M["CR"])):
            if isinstance(M["CR"][x], str):
                M["CR"][x] = trimES(M["CR"][x].strip())
        # M["CR"] = (trimES(M["CR"]).strip())  # remove foreign characters from CR (i.e. Chinese, Russian characters)

    if dbsource != "cochrane":
        M["AU"] = M["AU"].str.replace(u"\u2019",
                                      u"'")  # replace the right single quotation mark with the apostrophe character

    print("Done!\n")

    if dbsource not in ["dimensions", "pubmed", "lens"]:
        # AU_UN field creation
        if "C1" in M.columns:
            print("Generating affiliation field tag AU_UN from C1:  ")
            M = metaTagExtraction(M, Field="AU_UN")
            print("Done!\n")
        else:
            M["C1"] = np.nan
            M["AU_UN"] = np.nan

        # AU normalization
        M["AU"] = M["AU"].str.split(";").apply(lambda x: [re.sub("[^[:alnum:][-]']", " ", i).strip() for i in x])
        M["AU"] = M["AU"].apply(lambda x: ";".join([i for i in x if i]))

    if dbsource == "pubmed" and format == "pubmed":
        if "C1" in M.columns:
            print("Generating affiliation field tag AU_UN from C1:  ")
            M = metaTagExtraction(M, Field="AU_UN")
            print("Done!\n")
        else:
            M["C1"] = np.nan
            M["AU_UN"] = np.nan

    # SR field creation
    M = metaTagExtraction(M, Field="SR")
    d = M.duplicated(subset=["SR"])
    if d.sum() > 0:
        print(f"Removed {d.sum()} duplicated documents")
    M.drop_duplicates(subset=["SR"], inplace=True)
    M.set_index("SR", inplace=True)

    # bibliometrix>DB class
    # M.__class__ = ["bibliometrixDB", "pandas.core.frame.DataFrame"]

    return M