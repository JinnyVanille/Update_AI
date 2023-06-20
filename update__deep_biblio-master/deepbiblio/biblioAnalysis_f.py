# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd
import numpy as np
# from collections import Counter
# import re
import datetime as dt

from deepbiblio.metaTagExtraction_f import metaTagExtraction
from deepbiblio.tableTag_f import tableTag

def biblioAnalysis(M, sep=";"):
    # initialize variables
    Authors = None
    Authors_frac = None
    FirstAuthors = None
    PY = None
    FAffiliation = None
    Affiliation = None
    Affiliation_frac = None
    CO = np.repeat(np.nan, M.shape[0])
    TC = None
    TCperYear = None
    SO = None
    Country = None
    DE = None
    ID = None
    MostCitedPapers = None

    # M is the bibliographic dataframe
    Tags = list(M.columns)

    if "SR" not in Tags:
        M = metaTagExtraction(M, "SR")

    # temporal analysis
    if "PY" in Tags:
        PY = pd.to_numeric(M["PY"], errors="coerce")

    # Author's distribution
    if "AU" in Tags:
        listAU = M["AU"].str.split(sep)
        listAU = listAU.apply(lambda l: [x.strip() for x in l])
        nAU = listAU.apply(lambda l: len(l))  # num. of authors per paper
        fracAU = 1 / nAU  # fractional frequencies
        AU = listAU.explode().values

        Authors = dict(sorted(pd.Series(AU).value_counts().items(), key=lambda x: x[1], reverse=True))
        Authors_frac = pd.DataFrame({"Author": AU, "Frequency": fracAU}).groupby("Author").sum()
        Authors_frac = Authors_frac.sort_values(by="Frequency", ascending=False)
        FirstAuthors = listAU.apply(lambda l: l[0] if len(l) > 0 else np.nan)

        AuSingleAuthoredArt = len(set(FirstAuthors[nAU == 1]))
        AuMultiAuthoredArt = len(Authors) - AuSingleAuthoredArt

    # Total Citation Distribution
    if "TC" in Tags:
        TC = pd.to_numeric(M["TC"], errors="coerce")
        CurrentYear = dt.datetime.now().year
        TCperYear = TC / (CurrentYear - PY + 1)
        if "DI" not in Tags:
            M["DI"] = ""
        MostCitedPapers = pd.DataFrame({"Paper": M["SR"], "DOI": M["DI"], "TC": TC, "TCperYear": TCperYear, "PY": PY})
        MostCitedPapers["NTC"] = MostCitedPapers.groupby("PY")["TC"].transform(lambda x: x / x.mean())
        MostCitedPapers = MostCitedPapers.drop(columns=["PY"]).sort_values(by="TC", ascending=False)

    # ID Keywords
    if "ID" in M.columns:
        ID = tableTag(M, "ID", sep)

    # DE Keywords
    if "DE" in M.columns:
        DE = tableTag(M, "DE", sep)

    # Sources
    if "SO" in M.columns:
        SO = M["SO"].str.replace(",", "", regex=False)
        SO = SO.value_counts().sort_values(ascending=False)

    # All Affiliations, First Affiliation and Countries
    if ("C1" in M.columns) and (M["C1"].notna().sum() > 0):
        if not ("AU_UN" in M.columns):
            M = metaTagExtraction(M, Field="AU_UN")
        AFF = M["AU_UN"]
        listAFF = [x.split(sep) for x in AFF]
        nAFF = [len(x) for x in listAFF]
        listAFF = ['NA' if x == 0 else x for x in listAFF]
        fracAFF = [1 / x if x > 0 else 0 for x in nAFF]
        AFF = [x.lstrip() for y in listAFF for x in y]
        Affiliation = pd.Series(AFF).value_counts().sort_values(ascending=False)
        Affiliation_frac = pd.DataFrame({'Affiliation': AFF, 'Frequency': fracAFF}).groupby('Affiliation').sum()
        Affiliation_frac = Affiliation_frac.sort_values(by='Frequency', ascending=False)

        # First Affiliation
        FAffiliation = [x[0] for x in listAFF]

        # Countries
        countries = pd.read_csv("countries.csv")['name'].tolist()

        ### new code{
        if not ("AU1_CO" in M.columns):
            M = metaTagExtraction(M, Field="AU1_CO", sep=sep)
        CO = M["AU1_CO"]

        Country = tableTag(M, "AU1_CO")

        SCP_MCP = countryCollaboration(M, Country, k=len(Country), sep=sep)

    else:
        M["AU1_CO"] = np.nan
        SCP_MCP = pd.DataFrame({'Country': [np.nan], 'SCP': [np.nan]})

    if "DT" in M.columns:
        Documents = pd.Series(M["DT"]).value_counts()
        n = max([len(x) for x in Documents.index])
        Documents.index = [x.ljust(n + 5) for x in Documents.index]
    else:
        Documents = np.nan

    # international collaboration
    if not ("AU_CO" in M.columns):
        M = metaTagExtraction(M, Field="AU_CO", sep=sep)
    Coll = [len(set(x.split(sep))) > 1 for x in M["AU_CO"]]
    IntColl = sum(Coll) / M.shape[0] * 100

    Articles = M.shape[0]  # Articles
    Authors = M['AU'].str.split(sep, expand=True).stack().value_counts()  # Authors' frequency distribution
    AuthorsFrac = Authors / Authors.sum()  # Authors' frequency distribution (fractionalized)
    FirstAuthors = M['AF'].str.split(sep, expand=True)[0]  # First Author's list
    nAUperPaper = M['AU'].str.count(sep) + 1  # N. Authors per Paper
    Appearances = nAUperPaper.sum()  # Author appearances
    nAuthors = Authors.shape[0]  # N. of Authors
    AuMultiAuthoredArt = sum(nAUperPaper > 1)  # N. of Authors of multi-authored articles
    AuSingleAuthoredArt = sum(nAUperPaper == 1)  # N. of Authors of single-authored articles
    MostCitedPapers = M.sort_values(by='TC', ascending=False)  # Papers sorted by citations
    Years = M['PY']  # Years
    FirstAffiliation = M['FA'].str.split(sep, expand=True)[0]  # Affiliation of First Author
    Affiliations = M['AF'].str.split(sep, expand=True).stack().value_counts()  # Affiliations of all authors
    Aff_frac = Affiliations / Affiliations.sum()  # Affiliations of all authors (fractionalized)
    CO = M['CO']  # Country of each paper
    Countries = M['CO'].value_counts()  # Countries' frequency distribution
    CountryCollaboration = M.groupby('CO')['CO'].count()  # Intracountry (SCP) and intercountry (MCP) collaboration
    TotalCitation = M['TC'].sum()  # Total Citations
    TCperYear = TotalCitation / Years.max()  # Total Citations per year
    Sources = M['SO']  # Sources
    DE = M['DE'].str.split(sep, expand=True).stack().value_counts()  # Keywords
    ID = M['ID'].str.split(sep, expand=True).stack().value_counts()  # Authors' keywords
    Documents = M.shape[0]
    IntColl = M['Coll'].sum() / M.shape[0] * 100
    nReferences = M['NR'].sum()  # N. of References
    DB = M['DB'].iloc[0]

    results = {"Articles": Articles, "Authors": Authors, "AuthorsFrac": AuthorsFrac, "FirstAuthors": FirstAuthors,
               "nAUperPaper": nAUperPaper, "Appearances": Appearances, "nAuthors": nAuthors,
               "AuMultiAuthoredArt": AuMultiAuthoredArt, "AuSingleAuthoredArt": AuSingleAuthoredArt,
               "MostCitedPapers": MostCitedPapers, "Years": Years, "FirstAffiliation": FirstAffiliation,
               "Affiliations": Affiliations, "Aff_frac": Aff_frac, "CO": CO, "Countries": Countries,
               "CountryCollaboration": CountryCollaboration, "TotalCitation": TotalCitation, "TCperYear": TCperYear,
               "Sources": Sources, "DE": DE, "ID": ID, "Documents": Documents, "IntColl": IntColl,
               "nReferences": nReferences, "DB": DB}
    results = {"bibliometrix": results}

    return results



def countryCollaboration(M, Country, k, sep):
    if 'AU_CO' not in M.columns:
        M = metaTagExtraction(M, Field='AU_CO', sep=sep)

    M['SCP'] = 0
    M['SCP_CO'] = np.nan

    for i in range(M.shape[0]):
        if not pd.isna(M.loc[i, 'AU_CO']):
            co = M.loc[i, 'AU_CO']
            co = pd.Series(co.split(sep)).value_counts()

            if len(co) == 1:
                M.loc[i, 'SCP'] = 1

            M.loc[i, 'SCP_CO'] = M.loc[i, 'AU1_CO']
        else:
            M.loc[i, 'SCP'] = np.nan

    CO = list(Country.keys())[0:k]

    df = pd.DataFrame({'Country': [np.nan] * k, 'SCP': [0] * k})
    for i, co in enumerate(CO):
        df.loc[i, 'Country'] = co
        df.loc[i, 'SCP'] = M.loc[M['SCP_CO'] == co, 'SCP'].sum(skipna=True)

    df['MCP'] = tableTag(M, 'AU1_CO')[0:k] - df['SCP']
    return df