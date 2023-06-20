# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from functools import reduce
import numpy as np
import re

def dimensions2df(file, format="csv"):
    def replace_na(x):
        return x.fillna('') if is_string_dtype(x) else x

    def postprocessingDim(DATA):
        # code for post-processing of data goes here
        return DATA

    DATA = None
    bibtag = None

    if format == "csv":
        for i in range(len(file)):
            D = pd.read_csv(file[i], na_values="", quote='"', skiprows=1, engine='c')
            D = D.apply(replace_na)
            D = D.astype(str)
            if i > 0:
                l = list(set(l).intersection(D.columns))
                DATA = pd.concat([DATA[l], D[l]], ignore_index=True, sort=False)
            else:
                l = list(D.columns)
                DATA = D[l]

    elif format == "excel":
        for i in range(len(file)):
            D = pd.read_excel(file[i], skiprows=1, engine='openpyxl')
            D = D.apply(replace_na)
            D = D.astype(str)
            if i > 0:
                l = list(set(l).intersection(D.columns))
                DATA = pd.concat([DATA[l], D[l]], ignore_index=True, sort=False)
            else:
                l = list(D.columns)
                DATA = D[l]

    bibtag = pd.read_csv("bibtag.csv")  # replace with the path to the file

    DATA.columns = DATA.columns.str.replace('\\.|-', ' ')

    fields = DATA.columns.tolist()

    for i in range(len(fields)):
        ind = bibtag.loc[bibtag["DIMENSIONS"] == fields[i], "TAG"].index
        if not ind.empty:
            fields[i] = bibtag.loc[ind[0], "TAG"]

    DATA.columns = fields

    DATA = postprocessingDim(DATA)

    return DATA


def dimensions2df(file, format='csv'):
    def postprocessingDim(DATA):
        if 'Cited.references' in DATA.columns:
            def parse_cited_references(reference):
                reference = reference.replace('|', '!!!')
                reference = reference.split('!!!')

                ## first authors (of cited references)
                au = reference[0].split(';')
                au = [re.sub('^\\[|\\]$', '', x) for x in au]
                au = [re.sub('^,|(?<=,),|,$', '', x) for x in au]
                lastname = [re.sub(',.*', '', x).strip() for x in au]
                firstname = [re.sub('^[^,]*,', '', x).strip().split()[0] if ',' in x else '' for x in au]
                au = [' '.join([lastname[i], firstname[i]]).strip() for i in range(len(au))]

                ## publication year
                py = reference[3]
                so = reference[2]
                vol = 'V' + reference[4] if len(reference[4]) > 0 else ''
                num = 'N' + reference[5] if len(reference[5]) > 0 else ''

                ## doi
                doi = reference[7]

                ref = ', '.join([x for x in [', '.join([au, py, so, vol, num, doi]) if len(x) > 0 else '' for x in
                                             [au, py, so, vol, num, doi]] if len(x) > 0])
                return ref

            DATA['Cited.references'] = DATA['Cited.references'].astype(str)
            DATA['Cited.references'] = DATA['Cited.references'].apply(parse_cited_references)
        return DATA

    l = []
    for i in range(len(file)):
        if format == 'csv':
            D = pd.read_csv(file[i], na_values='', quote='"', skiprows=1)
        elif format == 'excel':
            D = pd.read_excel(file[i], skiprows=1)
        else:
            raise ValueError('Unsupported format')

        D = D.apply(lambda x: pd.to_numeric(x, errors='ignore') if np.issubdtype(x.dtype, np.number) else x)
        D = D.apply(lambda x: x.fillna('') if np.issubdtype(x.dtype, np.object) else x)

        if i > 0:
            l = list(set(l) & set(D.columns))
            DATA = pd.concat([DATA[l], D[l]])
        else:
            l = D.columns.tolist()
            DATA = D[l]

    bibtag = pd.read_csv('bibtag.csv')
    fields = [re.sub('\\.|-', ' ', x.strip()) for x in DATA.columns]
    fields = [bibtag[bibtag['DIMENSIONS'] == x]['TAG'].tolist()[0] if not bibtag[bibtag['DIMENSIONS'] == x][
        'TAG'].isnull().values.any() else x for x in fields]
    DATA.columns = fields

    DATA = postprocessingDim(DATA)
    # Converting original references in WOS format (AU, PY, SO, VOL, NUM, DOI)
    if "Cited.references" in DATA.columns:
        aaa = DATA["Cited.references"].str.split(";\\[")
        cr = aaa.apply(lambda l: ";".join([
            ",".join([
                re.sub("\\|", "!!!", x) for x in re.split("!!!", a)
            ]) for a in l
        ])).tolist()
        DATA["CR"] = [re.sub("] ];", "", c) for c in cr]
    else:
        DATA["CR"] = "NA,0000,NA"

    # Document Type
    if "DT" not in DATA.columns:
        DATA["DT"] = "Article"

    # Authors cleaning and converting in WoS format
    DATA["AF"] = DATA["AU"]
    DATA["AU"] = DATA["AU"].str.replace("\\s+", " ")
    DATA["AU"] = DATA["AU"].str.replace("\\(|\\)", "")
    listAU = DATA["AU"].str.split("; ")

    AU = []
    for l in listAU:
        lastname = re.sub(",.*", "", l)
        firstname = [x[0] for x in re.findall(r"(\w)\w*", re.sub(".*,", "", l))]
        AU.append(";".join([f"{ln} {fn}" for ln, fn in zip(lastname, firstname)]))

    DATA["AU"] = AU

    # Keywords
    if "DE" not in DATA.columns and "ID" not in DATA.columns:
        if "MeSH.terms" in DATA.columns:
            DATA["DE"] = DATA["ID"] = DATA["MeSH.terms"]
        else:
            DATA["DE"] = DATA["ID"] = "NA"
    if "DE" in DATA.columns and "ID" not in DATA.columns:
        DATA["ID"] = DATA["DE"]
    if "DE" not in DATA.columns and "ID" in DATA.columns:
        DATA["DE"] = DATA["ID"]

    # Affiliations
    DATA["RP"] = None
    if "AU_CO" not in DATA.columns:
        DATA["AU_CO"] = "NA"
        DATA["AU1_CO"] = "NA"
    else:
        DATA["AU1_CO"] = DATA["AU_CO"].str.split(";").apply(lambda l: l[0] if len(l) > 0 else "NA")

    i = np.where(np.array(list(DATA.keys())) == "Authors.Affiliations")[0]
    if len(i) == 1:
        DATA[list(DATA.keys())[i[0]]] = "AU_UN"

    if "AU_UN" in list(DATA.keys()):
        DATA["AU1_UN"] = [(l.split(";")[0]).strip() if len(l.split(";")) > 0 else np.nan for l in DATA["AU_UN"]]
    else:
        DATA["AU_UN"] = np.nan
        DATA["AU1_UN"] = np.nan

    DATA["AU1_CO"] = np.where(DATA["AU1_CO"] == "NA", np.nan, DATA["AU1_CO"])

    DATA["AU_CO"] = np.where(DATA["AU_CO"] == "NA", np.nan, DATA["AU_CO"])

    if "SO" in list(DATA.keys()) and "Anthology.title" in list(DATA.keys()):
        ind = np.where(pd.isnull(DATA["SO"]) | (DATA["SO"] == ""))[0]
        DATA.loc[ind, "SO"] = DATA.loc[ind, "Anthology.title"]
        DATA.loc[pd.isnull(DATA["SO"]) | (DATA["SO"] == "")] = "NA"

    if "SO" not in list(DATA.keys()):
        DATA["SO"] = "NA"

    ####
    print("\nCreating ISO Source names...")
    DATA["JI"] = DATA["SO"]
    # DATA$JI <- sapply(DATA$SO, AbbrevTitle, USE.NAMES = FALSE)
    DATA["J9"] = DATA["JI"].str.replace(".", "")
    ####

    DATA = DATA.apply(lambda x: x.str.upper() if isinstance(x, str) else x)

    DATA["PY"] = pd.to_numeric(DATA["PY"], errors="coerce")

    DATA["TC"] = pd.to_numeric(DATA["TC"], errors="coerce")

    DATA["DB"] = "ISI"
    return DATA
