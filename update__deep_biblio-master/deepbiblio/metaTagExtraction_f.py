# author:  Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

import re
import pandas as pd
import csv

def metaTagExtraction(M, Field='CR_AU', sep=';', aff_disamb=True):
    # data cleaning
    if 'CR' in M.columns:
        M['CR'] = M['CR'].str.replace('DOI;', 'DOI ')

    # SR field creation
    if Field == 'SR':
        M = SR(M)

    # CR_AU field creation
    if Field == 'CR_AU':
        M = CR_AU(M, sep)

    # CR_SO field creation
    if Field == 'CR_SO':
        M = CR_SO(M, sep)

    # AU_CO field creation
    if Field == 'AU_CO':
        M = AU_CO(M)

    # AU1_CO field creation
    if Field == 'AU1_CO':
        M = AU1_CO(M, sep)

    # UNIVERSITY AFFILIATION OF ALL AUTHORS AND CORRESPONDING AUTHOR
    if Field == 'AU_UN':
        # with disambiguation
        if aff_disamb:
            M = AU_UN(M, sep)
        else:
            # without disambiguation
            M['AU_UN'] = M['C1'].str.replace('\\[.*?\\] ', '')
            M['AU1_UN'] = M['RP'].str.split(sep).apply(lambda l: l[0])
            ind = M['AU1_UN'].str.find('),')
            a = ind[ind > -1].index
            M.loc[a, 'AU1_UN'] = M.loc[a, 'AU1_UN'].str[ind[a] + 2:]

    return M

def SR(M):
    listAU = [x.split(';') for x in M['AU'].astype(str)]
    listAU = [list(map(str.strip, x)) for x in listAU]
    if M['DB'][0] == 'scopus':
        listAU = [','.join(filter(None, x)) for x in listAU]
    else:
        listAU = [','.join(x) for x in listAU]

    FirstAuthors = [x.split(',')[0] if x != 'nan' and len(x.split(',')) > 0 else 'NA' for x in listAU]

    if 'J9' in M.columns and not M['J9'].isnull().all():
        no_art = M['J9'].isnull() & M['JI'].isnull()
        M.loc[no_art, 'J9'] = M.loc[no_art, 'SO']
        ind = M['J9'].isnull()
        M.loc[ind, 'J9'] = M.loc[ind, 'JI'].str.replace('.', ' ').str.strip()
        SR = ['{0}, {1}, {2}'.format(x, y, z) for x, y, z in zip(FirstAuthors, M['PY'], M['J9'])]
    else:
        no_art = M['JI'].isnull()
        M.loc[no_art, 'JI'] = M.loc[no_art, 'SO']
        J9 = M['JI'].str.replace('.', ' ').str.strip()
        SR = ['{0}, {1}, {2}'.format(x, y, z) for x, y, z in zip(FirstAuthors, M['PY'], J9)]

    M['SR_FULL'] = [re.sub('\s+', ' ', x) for x in SR]

    SR = [re.sub('\s+', ' ', x) for x in SR]
    st, i = 0, 0
    # while st == 0:
    #     ind = [j for j, x in enumerate(SR) if SR.count(x) > 1]
    #     if len(ind) > 0:
    #         i += 1
    #         for j in ind:
    #             SR[j] = '{0}-{1}'.format(SR[j], chr(96 + i))
    #     else:
    #         st = 1
    M['SR'] = SR

    return M


def CR_AU(M, sep):
    FCAU = []
    CCR = []
    CR = M['CR']
    listCAU = [x.split(sep) for x in CR]
    listCAU = [[l for l in sublist if len(l) > 10] for sublist in listCAU]  # delete not congruent references

    # vector of cited authors
    for i in range(len(M)):
        FCAU.append(re.sub("[[:punct:]]", "", listCAU[i][0].split(',')[0]).strip())
        CCR.append(';'.join(FCAU))

    M['CR_AU'] = CCR
    return M


def CR_SO(M, sep):
    FCAU = [[] for _ in range(len(M))]
    CCR = [None] * len(M)
    CR = M["CR"]
    listCAU = [s.split(sep) for s in CR.astype(str)]

    if M["DB"][1] == "ISI":
        for i in range(len(M)):
            elem = listCAU[i]
            ind = [len(e.split(",")) for e in elem]
            if max(ind) > 2:
                elem = [e for e in elem if len(e.split(",")) > 2]
                FCAU[i] = [e.strip() for e in [e.split(",")[2] for e in elem]]
                CCR[i] = ";".join(FCAU[i])
            else:
                CCR[i] = None

    elif M["DB"][1] == "SCOPUS":
        for i in range(len(M)):
            elem = [re.sub(".*?\\) ", "", s) for s in listCAU[i]]
            ind = [len(e.split(",")) for e in elem]
            if len(ind) > 0 and max(ind) > 2:
                elem = [e for e in elem if len(e.split(",")) > 2]
                FCAU[i] = [e.strip() for e in [e.split(",")[0] for e in elem]]
                CCR[i] = ";".join(FCAU[i])

    M["CR_SO"] = CCR
    return M

def AU_CO(M):
    # Countries
    size = M.shape[0]
    with open('./data/countries.csv', newline='') as f:
        reader = csv.reader(f)
        countries = list(reader)
    # countries = list(data("countries"))
    countries = [str(countries[0][i]) for i in range(len(countries[0]))]

    if M["DB"][0] in ["ISI", "PUBMED"]:
        countries = ["".join([s, "."]) for s in countries]
    elif M["DB"][0] == "SCOPUS":
        countries = ["".join([s, ";"]) for s in countries]

    M["AU_CO"] = None
    C1 = M["C1"].values

    if "RP" in M.columns:
        C1[pd.isna(C1)] = M.loc[pd.isna(C1), "RP"].values
    else:
        M["RP"] = None

    C1 = [re.sub(r"\[.*?\] ", "", s) if not pd.isna(s) else "NA" for s in C1]
    C1 = [s.rstrip(";") + "." if M["DB"][0] == "ISI" else s.rstrip(";") + ";" if M["DB"][0] == "SCOPUS" else s for s in
          C1]

    RP = M["RP"].values
    RP = [str(r) + ";" if not pd.isna(r) else "" for r in RP]

    for i in range(size):
        if not pd.isna(C1[i]):
            ind = [m.start() for l in countries for m in re.finditer(re.escape(l), C1[i])]
            if len(ind) > 0:
                M.at[i, "AU_CO"] = ";".join(list(set([countries[j].strip(";") for j in ind])))

        if pd.isna(M.at[i, "AU_CO"]):
            ind = [m.start() for l in countries for m in re.finditer(re.escape(l), RP[i])]
            if len(ind) > 0:
                M.at[i, "AU_CO"] = ";".join(list(set([countries[j].strip(";") for j in ind])))

    M["AU_CO"] = re.sub(r"\d+", "", M["AU_CO"])
    M["AU_CO"] = M["AU_CO"].replace(".", "")
    M["AU_CO"] = M["AU_CO"].replace(";;", ";")
    M["AU_CO"] = M["AU_CO"].replace("UNITED STATES", "USA")
    M["AU_CO"] = M["AU_CO"].replace("RUSSIAN FEDERATION", "RUSSIA")
    M["AU_CO"] = M["AU_CO"].replace("TAIWAN", "CHINA")
    M["AU_CO"] = M["AU_CO"].replace("ENGLAND", "UNITED KINGDOM")
    M["AU_CO"] = M["AU_CO"].replace("SCOTLAND", "UNITED KINGDOM")
    M["AU_CO"] = M["AU_CO"].replace("WALES", "UNITED KINGDOM")
    M["AU_CO"] = M["AU_CO"].replace("NORTH IRELAND", "UNITED KINGDOM")

    if M["DB"][0] == "ISI":
        M["AU_CO"] = remove_last_char(M["AU_CO"], last=".")
    if M["DB"][0] == "SCOPUS":
        M["AU_CO"] = remove_last_char(M["AU_CO"], last=";")

    return M

def AU1_CO(M, sep):
    size = M.shape[0]

    # Countries
    countries = pd.read_csv('./data/countries.csv', header=None)[0].astype(str)
    countries = countries.str.replace('\s+', ' ', regex=True)
    countries = ' ' + countries + ' '

    M['AU1_CO'] = None
    C1 = M['C1'].copy()
    C1[~M['RP'].isna()] = M.loc[~M['RP'].isna(), 'RP']
    C1 = C1.str.replace('\[.*?\] ', '', regex=True)
    C1 = C1.str.replace(r'^.*?\(REPRINT\sAUTHOR\)', '', regex=True)
    C1 = C1.str.split(sep).str[0]
    C1 = C1.str.replace(r'^.+?,', '', regex=True)
    C1 = C1.str.replace('[^\w\s]|_', ' ', regex=True)
    C1 = C1.str.strip()
    C1 = ' ' + C1 + ' '

    if M['DB'][0] != 'PUBMED':
        RP = M['RP'].copy()
        RP = RP.str.cat(sep=';')
        RP = RP.str.replace('[^\w\s]|_', ' ', regex=True)
        RP = RP.str.replace('\s+', ' ', regex=True)
    else:
        RP = C1.str.replace('[^\w\s]|_', ' ', regex=True)
        RP = ' ' + RP + ' '

    for i in range(size):
        if pd.notna(C1[i]):
            ind = countries.apply(lambda l: [m.start() for m in re.finditer(l, C1[i], flags=re.IGNORECASE)])
            ind = [x for sublist in ind for x in sublist if x >= 0]
            if len(ind) > 0:
                M.loc[i, 'AU1_CO'] = ';'.join(countries.loc[ind].apply(lambda l: l.strip()))

        if pd.isna(M.loc[i, 'AU1_CO']):
            ind = countries.apply(lambda l: [m.start() for m in re.finditer(l, RP[i], flags=re.IGNORECASE)])
            ind = [x for sublist in ind for x in sublist if x >= 0]
            if len(ind) > 0:
                M.loc[i, 'AU1_CO'] = ';'.join(countries.loc[ind].apply(lambda l: l.strip()))
    M['AU1_CO'] = M['AU1_CO'].str.replace(r'\d+', '', regex=True).str.strip()
    M['AU1_CO'] = M['AU1_CO'].str.replace('UNITED STATES', 'USA')
    M['AU1_CO'] = M['AU1_CO'].str.replace('RUSSIAN FEDERATION', 'RUSSIA')
    M['AU1_CO'] = M['AU1_CO'].str.replace('TAIWAN', 'CHINA')
    M['AU1_CO'] = M['AU1_CO'].str.replace('ENGLAND', 'UNITED KINGDOM')
    M['AU1_CO'] = M['AU1_CO'].str.replace('SCOTLAND', 'UNITED KINGDOM')
    M['AU1_CO'] = M['AU1_CO'].str.replace('WALES', 'UNITED KINGDOM')
    M['AU1_CO'] = M['AU1_CO'].str.replace('NORTH IRELAND', 'UNITED KINGDOM')
    # M['AU1_CO'] = M['AU1_CO'].str.replace('.', '', regex=False)
    # M['AU1_CO'] = M['AU1_CO'].str.replace(';;', ';', regex=False)
    return M


def AU_UN(M, sep):
    # remove reprint information from C1
    C1 = M["C1"]
    if "RP" not in M:
        M["RP"] = None

    for i in range(len(C1)):
        C1[i] = re.sub(r"\[.*?\]\s*", "", C1[i])
    AFF = C1;
    indna = [i for i in range(len(AFF)) if AFF[i] is None]
    if len(indna) > 0:
        for i in indna:
            AFF[i] = M["RP"][i]

    nc = [len(x) if x is not None else 0 for x in AFF]
    AFF = [x if nc[i] > 0 else None for i, x in enumerate(AFF)]

    listAFF = [re.split(sep, x, flags=re.IGNORECASE) if x is not None else None for x in AFF]

    uTags = ["UNIV", "COLL", "SCH", "INST", "ACAD", "ECOLE", "CTR", "SCI", "CENTRE", "CENTER", "CENTRO", "HOSP",
             "ASSOC", "COUNCIL", "FONDAZ", "FOUNDAT", "ISTIT", "LAB", "TECH", "RES", "CNR", "ARCH", "SCUOLA",
             "PATENT OFF",
             "CENT LIB", "HEALTH", "NATL", "LIBRAR", "CLIN", "FDN", "OECD", "FAC", "WORLD BANK", "POLITECN",
             "INT MONETARY FUND",
             "CLIMA", "METEOR", "OFFICE", "ENVIR", "CONSORTIUM", "OBSERVAT", "AGRI", "MIT ", "INFN", "SUNY "]

    AFFL = []
    for l in listAFF:
        if l is None:
            AFFL.append(None)
        else:
            l = [re.sub(r"\(REPRINT AUTHOR\)", "", x) for x in l]
            index = []
            for i in range(len(l)):
                affL = re.split(",", l[i], flags=re.IGNORECASE)
                indd = [j for j, x in enumerate(affL) for uTag in uTags if uTag in x.upper()]
                if len(indd) == 0:
                    index.append("NOTREPORTED")
                else:
                    index_affL = ND(affL, indd)
                    if index_affL["cond"]:
                        index.append("NOTDECLARED")
                    else:
                        index.append(index_affL["affL"])
            x = ""
            for i in range(len(index)):
                if isinstance(index[i], str):
                    x = index[i].strip() + ";" + x
                    # x = ";".join([x.strip() for x in index])
            x = x.replace(" ,", ";")
            AFFL.append(x)

    M["AU_UN"] = AFFL
    M["AU_UN"] = [x.replace("\\&", "AND").replace("&", "AND") if x is not None else None for x in M["AU_UN"]]

    # identification of Corresponding author affiliation
    RP = M['RP']
    RP[pd.isna(RP)] = M['C1'][pd.isna(RP)]
    AFF = [re.sub(r'\[.*?\] ', '', x) for x in RP]
    indna = [i for i, x in enumerate(AFF) if pd.isna(x)]
    if len(indna) > 0:
        for i in indna:
            AFF[i] = M['RP'][i]
    nc = [len(x) for x in AFF]
    AFF = [x if nc[i] > 0 else pd.NA for i, x in enumerate(AFF)]
    listAFF = [x.split(sep) for x in AFF if pd.notna(x)]
    AFFL = []
    for l in listAFF:
        l = [re.sub(r'\(REPRINT AUTHOR\)', '', x) for x in l]
        index = []
        for i in range(len(l)):
            affL = l[i].split(',')
            indd = [j for j, x in enumerate(affL) if any(tag in x for tag in uTags)]
            if len(indd) == 0:
                index.append("NOTREPORTED")
            elif any(re.search(r'\d', affL[x]) for x in indd):
                index.append("NOTDECLARED")
            else:
                index.append(affL[indd[0]])
        x = ""
        for i in range(len(index)):
            if isinstance(index[i], str):
                x = index[i].strip() + ";" + x
        # x = ';'.join((index).strip())
        x = re.sub(r' ,', ';', x)
        AFFL.append(x)
    M['AU1_UN'] = AFFL
    M['AU1_UN'] = [re.sub(r'\\\\&', 'AND', x) for x in M['AU1_UN']]
    M['AU1_UN'] = [re.sub(r'\\&', 'AND', x) for x in M['AU1_UN']]

    # identification of NR affiliations
    M['AU_UN_NR'] = pd.NA
    listAFF2 = [x.split(sep) for x in M['AU_UN'] if pd.notna(x)]
    for i in range(len(listAFF2)):
        cont = [j for j, x in enumerate(listAFF2[i]) if x == "NR"]
        if len(cont) > 0:
            M['AU_UN_NR'][i] = ';'.join(([listAFF[i][x] for x in cont]).strip())
    M['AU_UN'] = [x if pd.notna(AFF[i]) else pd.NA for i, x in enumerate(M['AU_UN'])]
    M['AU_UN'] = [x if x not in ["NOTDECLARED", "NOTREPORTED"] else pd.NA for x in M['AU_UN']]
    M['AU_UN'] = [re.sub(r'NOTREPORTED;', '', x) for x in M['AU_UN']]
    M['AU_UN'] = [re.sub(r';NOTREPORTED', '', x) for x in M['AU_UN']]
    M['AU_UN'] = [re.sub(r'NOTDECLARED;', '', x) for x in M['AU_UN']]
    M['AU_UN'] = [re.sub(r'NOTDECLARED', '', x) for x in M['AU_UN']]
    return M


def ND(affL, indd):
    if not indd:
        return {"cond": False, "affL": None}

    for i in indd:
        if not re.search(r"\d", affL[i]):
            return {"cond": False, "affL": affL[i]}

    return {"cond": True, "affL": None}

# lastChar function
def lastChar(C, last="."):
    A = C.str[-1]
    ind = (A != last) & (~A.isna())
    C[ind] = C[ind].str.cat([last] * ind.sum())
    return C

# removeLastChar function
def remove_last_char(C, last="."):
    A = C.str[-1]
    ind = (A == last) & (~A.isna())
    C[ind] = C[ind].str[:-1]
    return C

# ND function
def ND(affL, indd):
    # aff = affL[~affL.str.contains(r"\d")]
    # ind = indd[~affL.str.contains(r"\d")]
    cond = len(indd) < 1
    # r = {"affL": affL.iloc[indd[0]], "cond": cond}
    r = {"affL": affL, "cond": cond}
    return r