# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

import pandas as pd
import re
import pyreadr

from deepbiblio.AbbrevTitle_f import AbbrevTitle
from deepbiblio.trimES_f import trimES


def removeStrangeChar(D):
    # implement function to remove strange characters
    return D


def bib2df(D, dbsource="isi"):
    # D <- D[nchar(D)>1]

    # remove empty rows and strange characters
    try:
        D = D[pd.Series(D).str.len() > 1]
    except:
        D = removeStrangeChar(D)
        # next

    D = re.sub("\\{\\[\\}", "[", D)
    D = re.sub("\\{\\]\\}", "]", D)
    Papers = [i for i, s in enumerate(D) if s.startswith("@")]  # first row of each document
    if (Papers[0] > 1):
        D = D[Papers[0] - 1:]
        Papers = [i - (Papers[0] - 1) for i in Papers]

    if (dbsource == "isi"): D = re.sub(" = \\{", "={", D)

    D = re.sub("\\\t", "", re.sub(" = \\{", "={", D))  # to work also with new scopus bib format

    # D[Papers] = ["Paper={" + s for s in D[Papers]]
    for i in range(len(Papers)):
        Papers[i] = ["Paper={" + str(Papers[i]) + "}"]

    ind = [m.start() for m in re.finditer("=\\{", D)]  # sep among tags and contents
    # ind[Papers] = 6

    nP = len(Papers)  # number of documents

    # for i in range(len(D)):
    #     if (ind[i] == -1):
    #         D[i] = trimES(D[i - 1][:ind[i - 1] + 1] + D[i])
    #         ind[i] = ind[i - 1]

    # rowPapers = [j - i for i, j in zip(Papers, Papers[1:] + [len(D) + 1])]
    #
    # numPapers = [i + 1 for i in range(nP) for j in range(rowPapers[i])]
    rowPapers = len(Papers)
    numPapers = nP

    DATA = pd.DataFrame({
        'Tag': [s[:ind[i] + 1] for i, s in enumerate(D)],
        'content': [s[ind[i] + 2:] for i, s in enumerate(D)],
        'Paper': numPapers
    })
    DATA['content'] = DATA['content'].str.replace("\\}|\\},|\\{", "")

    df = (DATA.groupby(['Paper', 'Tag'])
          .agg(cont=('content', lambda x: '---'.join(x)))
          .reset_index()
          .pivot(index='Paper', columns='Tag', values='cont')
          .reset_index()
          )

    df = df.copy()

    del DATA
    bibtag = None
    # bibtag = data("bibtag", envir=environment())
    bibtag = pyreadr.read_r('./data/bibtag.rda')
    bibtag = pd.DataFrame(bibtag)

    Tag = [x.lower() for x in df.columns]
    if dbsource == "scopus":
        bibtag = bibtag[(bibtag["SCOPUS"].isin(Tag))]
        for i in range(bibtag.shape[0]):
            Tag = [bibtag.iloc[i]['TAG'] if x == bibtag.iloc[i]['SCOPUS'] else x for x in Tag]
    elif dbsource == "isi":
        bibtag = bibtag[(bibtag["ISI"].isin(Tag))]
        for i in range(bibtag.shape[0]):
            Tag = [bibtag.iloc[i]['TAG'] if x == bibtag.iloc[i]['ISI'] else x for x in Tag]
    elif dbsource == "generic":
        bibtag = bibtag[(bibtag["GENERIC"].isin(Tag))]
        for i in range(bibtag.shape[0]):
            Tag = [bibtag.iloc[i]['TAG'] if x == bibtag.iloc[i]['GENERIC'] else x for x in Tag]

    df.columns = [x.replace("={}", "") for x in Tag]

    ### replace "---" with ";"
    tagsComma = ["AU", "DE", "ID", "C1", "CR"]
    nolab = list(set(tagsComma) - set(df.columns))
    if len(nolab) > 0:
        print(
            "\nWarning:\nIn your file, some mandatory metadata are missing. Bibliometrix functions may not work properly!\nPlease, take a look at the vignettes:\n- 'Data Importing and Converting' (https://www.bibliometrix.org/vignettes/Data-Importing-and-Converting.html)\n- 'A brief introduction to bibliometrix' (https://www.bibliometrix.org/vignettes/Introduction_to_bibliometrix.html)\n\n")
        print("\nMissing fields: ", nolab, "\n")

    tagsComma = list(filter(lambda x: x not in nolab, tagsComma))
    df1 = df[tagsComma].applymap(lambda x: x.replace("---", ";"))

    ### replace "---" with " "
    otherTags = list(set(df.columns) - set(tagsComma))
    df2 = df[otherTags].applymap(lambda x: trimES(x.replace("---", " ")))
    df = pd.concat([df1, df2], axis=1)

    # Funding info
    ind = [i for i, x in enumerate(df.columns) if "funding_text" in x]
    if ("FX" not in df.columns) and len(ind) > 0:
        df["FX"] = df.iloc[:, ind].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        df.drop(df.columns[ind], axis=1, inplace=True)

    df = postprocessing(df, dbsource)

    df = df.loc[:, df.columns != "Paper"]
    df = df.loc[:, df.columns != "paper"]
    return df

def postprocessing(DATA, dbsource):
    # Authors' names cleaning (surname and initials)
    # remove ; and 2 or more spaces
    DATA['AU'] = DATA['AU'].str.replace('\s{2,}', ' ')

    listAU = DATA['AU'].str.split(" and ")

    AU = list(map(lambda l: ';'.join(
        [re.sub(",.*", "", x.strip()) + " " + ''.join(re.findall(r'\b\w', x.strip())) for x in l]), listAU))

    DATA['AU'] = AU

    # TC post-processing
    if 'TC' in DATA.columns:
        DATA['TC'] = pd.to_numeric(DATA['TC'].str.extract('(\d+)'), errors='coerce')

    # CR post-processing
    if 'CR' in DATA.columns:
        # remove dots after DOIs
        DATA['CR'] = DATA['CR'].str.replace('\.;', ';')
        DATA['CR'] = DATA['CR'].str.slice(stop=-1)

    # Year
    if 'PY' in DATA.columns:
        DATA['PY'] = pd.to_numeric(DATA['PY'].str.extract('(\d+)'), errors='coerce')

    if 'UT' in DATA.columns:
        DATA['UT'] = DATA['UT'].str.replace(':', '')

    if 'RP' not in DATA.columns and 'C1' in DATA.columns:
        DATA['RP'] = DATA['C1'].str.split('.').str[0]

    # keywords post-processing (missing ";" in some rows)
    if 'ID' in DATA.columns:
        DATA['ID'] = DATA['ID'].str.replace('   |,', ';')

    if 'DE' in DATA.columns:
        DATA['DE'] = DATA['DE'].str.replace('   |,', ';')

    ### merge Sources and Proceedings
    if ('SO' in DATA.columns) and ('BO' in DATA.columns):
        ind = DATA['SO'].isna()
        DATA.loc[ind, 'SO'] = DATA.loc[ind, 'BO']

    if 'PN' in DATA.columns:
        DATA['PN'] = pd.to_numeric(DATA['PN'].str.extract('(\d+)'), errors='coerce')

    if dbsource != 'generic':
        DATA['DB'] = dbsource
    else:
        DATA['DB'] = 'SCOPUS'

    # Toupper
    DI = DATA['DI']
    URL = DATA['url']
    DATA = DATA.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    if 'JI' in DATA.columns:
        DATA['J9'] = DATA['JI'].str.replace('\.', '')
    else:
        DATA['J9'] = DATA['JI'].apply(lambda x: AbbrevTitle(x) if isinstance(x, str) else None)
        DATA['JI'] = DATA['J9']
    DATA['DI'] = DI
    DATA['url'] = URL


    return DATA

