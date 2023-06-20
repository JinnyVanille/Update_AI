# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

import pandas as pd
import numpy as np
import os
from pathlib import Path
from urllib.parse import urlparse
import requests


def csvLens2df(files):
    where = None

    # import all files in a single data frame
    for i, file in enumerate(files):
        D = pd.read_csv(file, na_values='', quotechar='"', engine='python')
        D = D.replace(np.nan, '', regex=True)
        D = D.astype(str)

        if i > 0:
            l = list(set(l).intersection(set(D.columns)))
            DATA = pd.concat([DATA[l], D[l]])
        else:
            l = list(D.columns)
            DATA = D

    DATA = relabelling(DATA)

    # Authors' names cleaning (surname and initials)
    DATA['AF'] = DATA['AU']
    listAU = DATA['AU'].str.split('; ')
    AU = []
    for l in listAU:
        lastname = l.str.extract('(\S+$)').iloc[0]
        firstname = l.str.extract('([A-Z]:?)+').str.replace(':', '').str.strip().iloc[0]
        AU.append(lastname + ', ' + firstname)
    DATA['AU'] = AU

    # Iso Source Titles
    DATA.loc[DATA['SO'] == '', 'SO'] = DATA.loc[DATA['SO'] == '', 'Publisher']
    DATA['JI'] = DATA['SO'].apply(AbbrevTitle)
    DATA['J9'] = DATA['JI'].str.replace('.', '')
    DATA['ID'] = DATA['DE']
    DI = DATA['DI']
    URL = DATA['URL']
    DATA = DATA.applymap(str.upper)
    DATA['DI'] = DI
    DATA['URL'] = URL
    DATA['AU_CO'] = 'NA'
    DATA['DB'] = 'LENS'

    return DATA


def AbbrevTitle(title):
    if title == '':
        return ''
    words = title.split()
    stopwords = ['AND', 'THE', 'OF', 'IN', 'FOR', 'TO', 'AN', 'A', 'BY', 'WITH', 'ON', 'AT']
    abbrev = ''.join([word[0] for word in words if word.upper() not in stopwords])
    return abbrev


import pandas as pd


def relabelling(DATA):
    # column re-labelling
    label = DATA.columns
    label = label.str.replace("Source Title", "SO")
    # label = label.str.replace("Authors with affiliations", "C1")
    label = label.str.replace("Author/s", "AU")
    label = label.str.replace("Publication.Type", "DT")
    label = label.str.replace("Title", "TI")
    label = label.str.replace("Publication Year", "PY")
    label = label.str.replace("Volume", "VL")
    label = label.str.replace("Issue Number", "IS")
    label = label.str.replace("Source Country", "SO_CO")
    label = label.str.replace("Scholarly Citation Count", "TC")
    label = label.str.replace("DOI", "DI")
    label = label.str.replace("Source URLs", "URL")
    label = label.str.replace("Abstract", "AB")
    label = label.str.replace("Keywords", "DE")
    label = label.str.replace("MeSH Terms", "MESH")
    label = label.str.replace("Funding Details", "FU")
    label = label.str.replace("Funding", "FX")
    label = label.str.replace("References", "CR")
    # label = label.str.replace("Correspondence Address", "RP")
    label = label.str.replace("Fields of Study", "SC")
    label = label.str.replace("Language of Original Document", "LA")
    label = label.str.replace("Document Type", "DT")
    label = label.str.replace("Source", "DB")
    label = label.str.replace("Lens ID", "UT")
    DATA.columns = label

    return DATA