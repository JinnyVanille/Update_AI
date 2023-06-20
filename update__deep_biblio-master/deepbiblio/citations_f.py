# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import re
from collections import Counter

def citations(M, field="article", sep=";"):
    CR = []
    Year = []
    SO = []
    listCR = [x.split(sep) for x in M['CR']]

    ## check for empty CR
    if sum([len(x) > 3 for x in listCR if x is not None]) == 0:
        print("\nReference metadata field 'CR' is empty!!\n\n")
        return None

    if field == "author":
        if M['DB'][0] == "ISI":
            listCR = [list(map(lambda x: x.split(',')[0].replace("[^[:alnum:]]", " ").strip(), x)) for x in listCR]
        if M['DB'][0] == "SCOPUS":
            listCR = [list(map(lambda x: re.split("\\., ", x)[
                                         :re.search("[[:digit:]]", x).start() if re.search("[[:digit:]]",
                                                                                           x) is not None else 1][
                0].replace("[^[:alnum:]]", " ").strip(), x)) for x in listCR]

    if field == "article":
        listCR = [list(filter(lambda x: "," in x, x)) for x in listCR]

    CR = [x for x in listCR if x is not None]
    CR = [x for x in CR if len(x) >= 3]
    CR = [x.lstrip() for x in CR]
    CR = {k: v for k, v in sorted(Counter(CR).items(), key=lambda x: x[1], reverse=True)}

    if field == "article":
        if M['DB'][0] == "ISI":
            listCR = [x.split(",") for x in CR.keys()]
            Year = [int(x[1]) if len(x) > 1 else None for x in listCR]
            SO = [x[2] if len(x) > 2 else None for x in listCR]
            SO = [x.lstrip() for x in SO]
        elif M['DB'][0] == "SCOPUS":
            REF = list(CR.keys())
            y = yearSoExtract(REF)
            Year = y['Year']
            SO = y['SO']

    return {"Cited": CR, "Year": Year, "Source": SO}

def yearSoExtract(string):
    # for Scopus references
    pattern = r'\(\d{4}\)'
    ind = [m.start() for m in re.finditer(pattern, string)]
    ind = [-1 if x is None else x for x in ind]
    string = [string[i:i+7] if i in ind else "(0000)" for i in range(len(string))]
    ind = [i if i != -1 else 1 for i in ind]
    for i in range(len(ind)):
        if ind[i] == -1:
            ind[i] = 1
            attr_match = [6]
            break
    else:
        attr_match = None
    y = [re.findall(r'\d{4}', string[i])[0] for i in ind]
    SO = [re.sub(r',.*$', '', string[i+7:]) for i in ind]
    return {'Year': y, 'SO': SO}