# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

import pandas as pd


def cochrane2df(D):
    D = list(filter(lambda x: len(x) > 0, D))  # remove empty rows

    Papers = [i for i in range(len(D)) if D[i][:8] == "Record #"]  # first row of each document
    nP = len(Papers)  # number of documents

    rowPapers = [Papers[i + 1] - Papers[i] for i in range(nP)]
    numPapers = [i for i, n in enumerate(rowPapers) for _ in range(n)]

    DATA = pd.DataFrame({"Tag": [d[:4] for d in D], "content": [d[4:].strip() for d in D], "Paper": numPapers})
    DATA["Tag"] = DATA["Tag"].str.replace(":", "").str.replace(" ", "")

    df = DATA.groupby(["Paper", "Tag"])["content"].apply(lambda x: " ; ".join(x)).reset_index()
    df = df.pivot(index="Paper", columns="Tag", values="content")
    df = df.rename(columns={"YR": "PY", "ID": "UT", "KY": "ID", "US": "URL", "DOI": "DI", "NO": "NR"})

    df["PY"] = pd.to_numeric(df["PY"], errors="coerce")

    ### replace ";" with "; "
    tagsComma = ["AU", "ID"]
    df1 = df[tagsComma].apply(lambda x: x.str.replace(" ; ", ";"))

    ### replace " " with "---"
    otherTags = df.columns.difference(tagsComma)
    df2 = df[otherTags].apply(lambda x: x.str.replace(" ", "---"))

    df = pd.concat([df1, df2], axis=1)
    del df1, df2

    df["ID"] = df["ID"].str.replace(r"\[[^\]]*\]", "", regex=True).str.replace("; ", ";").str.replace(" ;", ";")

    df["DB"] = "COCHRANE"

    DI = df["DI"]
    df = df.apply(lambda x: x.str.upper() if x.name != "DI" else x)
    df["DI"] = DI.str.replace(" ", "")

    df = df.drop(columns=["Paper"])
    df["DE"] = df["ID"]
    df["JI"] = df["J9"] = df["SO"]

    return df