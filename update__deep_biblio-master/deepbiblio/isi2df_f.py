# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd

def isi2df(D):
    # remove empty rows and strange characters
    D = [row for row in D if len(row) > 1]

    try:
        D = [row for row in D if row.encode('utf-8').isalnum()]
    except:
        pass

    D = [row for row in D if not row.startswith(('FN ', 'VR '))]

    for i in range(1, len(D)):
        if D[i][:3] == "   ":
            D[i] = D[i].replace(D[i][:3], D[i - 1][:3])

    Papers = [i for i, row in enumerate(D) if row.startswith('PT ')]  # first row of each document
    nP = len(Papers)  # number of documents

    rowPapers = [Papers[i + 1] - Papers[i] for i in range(len(Papers) - 1)]
    rowPapers.append(len(D) - Papers[-1])

    numPapers = [i + 1 for i in range(nP) for j in range(rowPapers[i])]

    DATA = pd.DataFrame({'Tag': [row[:3] for row in D], 'content': [row[4:].rstrip() for row in D], 'Paper': numPapers})
    DATA['Tag'] = DATA['Tag'].str.replace(' ', '')

    df = DATA.groupby(['Paper', 'Tag']).agg({'content': lambda x: '---'.join(x)}).reset_index()
    df = df.pivot(index='Paper', columns='Tag', values='content').reset_index()

    missingTags = set(['AU', 'DE', 'C1', 'RP', 'CR', 'PY', 'SO', 'TI', 'TC']).difference(df.columns)
    if len(missingTags) > 0:
        print(
            "\nWarning:\nIn your file, some mandatory metadata are missing. Bibliometrix functions may not work properly!\nPlease, take a look at the vignettes:\n- 'Data Importing and Converting' (https://www.bibliometrix.org/vignettes/Data-Importing-and-Converting.html)\n- 'A brief introduction to bibliometrix' (https://www.bibliometrix.org/vignettes/Introduction_to_bibliometrix.html)\n")
        print("Missing fields:", missingTags)

    df['PY'] = pd.to_numeric(df['PY'], errors='coerce')

    ### replace "---" with ";"
    tagsComma = ["AU", "AF", "CR"]

    nolab = list(set(tagsComma) - set(df.columns))

    tagsComma = [t for t in tagsComma if t not in nolab]

    df1 = df[tagsComma].apply(lambda x: x.str.replace("---", ";"))

    ### replace "---" with " "
    otherTags = list(set(df.columns) - set(tagsComma))
    df2 = df[otherTags].apply(lambda x: x.str.replace("---", " ").str.strip())
    df = pd.concat([df1, df2], axis=1)
    del df1, df2

    df["DB"] = "ISI"

    # Authors
    df["AU"] = df["AU"].str.replace(",", " ").str.strip()

    # Toupper
    DI = df["DI"]
    df = df.apply(lambda x: x.str.upper())
    df["DI"] = DI

    # add sep ; to affiliations
    df["C1"] = df["C1"].str.replace(r"\[.*?\]", "").str.strip()  # to remove author info in square brackets
    df["C1"] = df["C1"].str.replace(".", ";.")

    df = df[df.columns[df.columns != "Paper"]]
    return df
