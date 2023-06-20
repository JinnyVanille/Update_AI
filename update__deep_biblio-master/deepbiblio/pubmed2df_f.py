# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd
import numpy as np

def pubmed2df(D):
    # Remove empty rows
    D = D[D.str.len() > 0]

    # Replace indentation in the beginning of a line with the same as the previous line
    for i in range(1, len(D)):
        if D[i][:4] == "    ":
            D[i] = D[i - 1][:4] + D[i][4:]

    # Find the first row of each document
    Papers = np.where(np.char.find(D, "PMID-") == 0)[0]
    nP = len(Papers)  # number of documents

    # Calculate the number of rows in each document
    rowPapers = np.diff(np.concatenate((Papers, [len(D)])))

    # Create a column indicating the document number for each row
    numPapers = np.repeat(np.arange(1, nP + 1), rowPapers)

    # Create a dataframe with the data
    DATA = pd.DataFrame({'Tag': np.char.strip(D.str[:4]), 'content': D.str[7:], 'Paper': numPapers})

    # Combine rows with the same paper number and tag into one row, separating content with '---'
    df = DATA.groupby(['Paper', 'Tag'])['content'].apply(lambda x: '---'.join(x)).reset_index()

    # Pivot the dataframe to have tags as columns
    df = df.pivot(index='Paper', columns='Tag', values='content')

    # Rename field tags
    old_labs = ['AD', 'AUID', 'FAU', 'IS', 'IP', 'SO', 'JT', 'TA', 'MH', 'PG', 'PT', 'VI', 'DP']
    new_labs = ['C1', 'OI', 'AF', 'SN', 'IS', 'SO2', 'SO', 'J9', 'DE', 'PP', 'DT', 'VL', 'PY']
    df = df.rename(columns=dict(zip(old_labs, new_labs)))

    # if error == 1:
    #     print(
    #         "\nWarning:\nIn your file, some mandatory metadata are missing. Bibliometrix functions may not work properly!\nPlease, take a look at the vignettes:\n- 'Data Importing and Converting' (https://www.bibliometrix.org/vignettes/Data-Importing-and-Converting.html)\n- 'A brief introduction to bibliometrix' (https://www.bibliometrix.org/vignettes/Introduction_to_bibliometrix.html)\n\n")

    # extract DOIs
    df = pd.DataFrame(D)
    df.columns = df.columns.str.strip()
    df["DI"] = df["LID"].str.split("[", expand=True)[0].str.strip()
    df["PY"] = pd.to_numeric(df["PY"].str[:4], errors="coerce")

    ### replace "---" with ";"
    tagsComma = ["AU", "AF", "DE", "AID", "OT", "PHST", "DT"]
    nolab = set(df.columns) - set(tagsComma)
    tagsComma = [t for t in tagsComma if t not in nolab]

    df1 = df[tagsComma].apply(lambda x: x.str.replace("---", ";"))

    ### replace "---" with " "
    otherTags = set(df.columns) - set(tagsComma)
    df2 = df[otherTags].apply(lambda x: x.str.replace("---", " ").str.strip())
    df = pd.concat([df1, df2], axis=1)

    df["DB"] = "PUBMED"

    # remove * char from keywords
    df[["DE", "ID"]] = df[["DE", "ID"]].apply(lambda x: x.str.replace("\\*", ""))
    df = df.apply(lambda x: x.str.upper() if x.name in ["DE", "ID"] else x)

    # add sep ; to affiliations
    df["C1"] = df["C1"].str.replace("\\.", ".;", regex=False)
    df["RP"] = np.nan
    df = df.drop("Paper", axis=1)

    return df