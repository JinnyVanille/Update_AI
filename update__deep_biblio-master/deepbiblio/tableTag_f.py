# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd
import re

from deepbiblio.termExtraction_f import *


def tableTag(M, Tag="CR", sep=";", ngrams=1, remove_terms=None, synonyms=None):
    # Check and remove duplicates
    # M = M.drop_duplicates(subset=["SR"])

    if Tag in ["AB", "TI"]:
        M = termExtraction(M, Field=Tag, stemming=False, verbose=False, ngrams=ngrams, remove_terms=remove_terms, synonyms=synonyms)
        i = f"{Tag}_TM"
    else:
        i = Tag

    if Tag == "C1":
        M["C1"] = M["C1"].str.replace(r"\[.+?]", "", regex=True)

    # Split values and remove punctuations and extra spaces
    Tab = M[i].str.split(sep, expand=True).stack().str.strip().str.replace(r"\s+|\.", " ").reset_index(level=1, drop=True)

    # Merge synonyms in the vector synonyms
    if synonyms is not None and isinstance(synonyms, str):
        s = [re.split(";", x.upper()) for x in synonyms.split()]
        snew = [x[0].strip() for x in s]
        sold = [list(map(str.strip, x[1:])) for x in s]
        for i in range(len(s)):
            Tab.replace(to_replace=sold[i], value=snew[i], inplace=True)

    Tab = Tab.value_counts().sort_values(ascending=False)

    # Remove terms from ID and DE
    if Tag in ["DE", "ID"] and remove_terms is not None:
        term = set(Tab.index) - set(remove_terms.upper())
        Tab = Tab[term]

    return Tab