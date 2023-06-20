# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd
from deepbiblio import *
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
import pandas as pd
from pandas.api.types import CategoricalDtype

from deepbiblio.tableTag_f import *
from deepbiblio.trimES_f import trimES
import pyreadr

def termExtraction(M, Field="TI", ngrams=1, stemming=False, language="english", remove_numbers = True,
                   remove_terms=None, keep_terms=None, synonyms=None, verbose=True):

    # ngrams imposed = 1 for keywords
    if Field in ["ID", "DE"]:
        ngrams = 1

    # load stopwords
    stopwords = []
    stop_words = pyreadr.read_r("tidytext", "./data/stop_words.csv")
    stop_words = stop_words["word"]

    if ngrams == 2:
        remove_terms.extend(stopwords["bigrams"])

    if language == "english":
        stopwords = stop_words.values.tolist()
    # elif language == "italian":
    #     stopwords = pd.read_csv(pkg_resources.resource_filename("tm", "data/stopwords-it.txt"), header=None).values.tolist()
    # elif language == "german":
    #     stopwords = pd.read_csv(pkg_resources.resource_filename("tm", "data/stopwords-de.txt"), header=None).values.tolist()
    # elif language == "french":
    #     stopwords = pd.read_csv(pkg_resources.resource_filename("tm", "data/stopwords-fr.txt"), header=None).values.tolist()
    # elif language == "spanish":
    #     stopwords = pd.read_csv(pkg_resources.resource_filename("tm", "data/stopwords-es.txt"), header=None).values.tolist()

    stopwords = [word.lower() for word in stopwords]

    # remove all special characters (except "-" becoming "_")
    TERMS = M.loc[:, ["SR", Field]].rename(columns={Field: "text"})

    TERMS["text"] = TERMS["text"].str.replace(" - ", " ")

    # save original multi-words keywords
    if Field in ["ID", "DE"]:
        listTerms = TERMS["text"].str.split(";")
        TERMS["text"] = [re.sub("-", "__", trimES(re.sub(r"\s+", "_", l.strip()))) for l in listTerms]
    else:
        TERMS["text"] = TERMS["text"].str.replace("[^a-zA-Z0-9\s\-]", "").str.replace("-", "__").str.lower()

    # remove numbers
    if remove_numbers:
        TERMS["text"] = TERMS["text"].str.replace("\d+", "")

    # keep terms in the vector keep.terms
    if keep_terms is not None and isinstance(keep_terms, list):
        keep_terms = [term.lower() for term in keep_terms]
        if Field in ["DE", "ID"]:
            keep_terms = [re.sub(r"\s+", "_", term) for term in keep_terms]
            keep_terms = [re.sub("-", "__", term) for term in keep_terms]
        else:
            keep_terms = [re.sub("-", "__", term) for term in keep_terms]
        for i in range(len(keep_terms)):
            TERMS["text"] = TERMS["text"].str.replace(keep_terms[i], keep_terms[i])

    if remove_terms is None:
        remove_terms = ""

        # load stopwords
    stopwords = []
    if language == "english":
        stopwords = stopwords.words('english')
    elif language == "italian":
        stopwords = stopwords.words('italian')
    elif language == "german":
        stopwords = stopwords.words('german')
    elif language == "french":
        stopwords = stopwords.words('french')
    elif language == "spanish":
        stopwords = stopwords.words('spanish')

    # ngrams imposed = 1 for keywords
    if Field in ["ID", "DE"]:
        ngrams = 1

    # remove all special characters (except "-" becoming "_")
    TERMS = M.loc[:, ["SR", Field]]
    TERMS.columns = ["SR", "text"]
    TERMS["text"] = TERMS["text"].str.replace(" - ", " ")

    # save original multi-words keywords
    if Field in ["ID", "DE"]:
        list_terms = TERMS["text"].str.split(";")
        TERMS["text"] = list_terms.apply(
            lambda l: ";".join([re.sub("-", "__", l) for l in map(lambda x: "_".join(x.split()), l)]))
    else:
        TERMS["text"] = TERMS["text"].apply(lambda x: re.sub("[^a-zA-Z0-9\s\-]", "", x.lower()))

    # remove numbers
    if remove_numbers:
        TERMS["text"] = TERMS["text"].apply(lambda x: re.sub("[0-9]", "", x))

    # keep terms in the vector keep.terms
    if keep_terms is not None and isinstance(keep_terms, list):
        keep_terms = [term.lower() for term in keep_terms]
        if Field in ["DE", "ID"]:
            kt = [re.sub(" ", "_", term) for term in keep_terms]
            kt = [re.sub("-", "__", term) for term in kt]
        else:
            kt = [re.sub("-", "__", term) for term in keep_terms]
        for i in range(len(keep_terms)):
            TERMS["text"] = TERMS["text"].apply(lambda x: re.sub(keep_terms[i], kt[i], x))

    TERMS = extractNgrams(text=TERMS, Var="text", nword=ngrams, stopwords=stopwords, custom_stopwords=remove_terms.lower(),
                          stemming=stemming, language=language, synonyms=synonyms, Field=Field)

    TERMS = TERMS.loc[~TERMS["ngram"].str.contains("NA", na=False)].groupby("SR")["ngram"].apply(
        lambda x: ";".join(x)).reset_index(name="text")

    # assign the vector to the bibliographic data frame
    col_name = Field + "_TM"
    M = M.drop(columns=[col_name], errors="ignore")
    M = pd.merge(TERMS, M, on="SR", how="right")
    M = M.rename(columns={"text": col_name})

    # display results
    if verbose:
        s = tableTag_fn(M, col_name)
        if len(s) > 25:
            print(s.iloc[:25])
        else:
            print(s)

    M = M.set_index("SR")
    M = M.astype({"PY": int})
    M.index.name = None

    return M


def extractNgrams(text, Var, nword, stopwords, custom_stopwords, stemming, language, synonyms, Field):
    stopwords += ["elsevier", "springer", "wiley", "mdpi", "emerald", "originalityvalue", "designmethodologyapproach",
                  "-", " -", "-present", "-based", "-literature", "-matter"]
    custom_stopngrams = custom_stopwords + ["rights reserved", "john wiley", "john wiley sons", "science bv",
                                            "mdpi basel",
                                            "mdpi licensee", "emerald publishing", "taylor francis", "paper proposes",
                                            "we proposes", "paper aims", "articles published", "study aims",
                                            "research limitationsimplications"]
    ngram = None

    # ngrams <- text %>%
    #   drop_na(any_of(Var)) %>%
    #   unnest_tokens(ngram, !!Var, token = "ngrams", n = nword) %>%
    #   separate(.data$ngram, paste("word",1:nword,sep=""), sep = " ")
    ngrams = text.dropna(subset=[Var]).assign(ngram=text[Var].apply(lambda x:
                                                                    [tuple(x.split()[i:i + n]) for n in nword for i in
                                                                     range(len(x.split()) - n + 1)])).explode("ngram")
    ngrams = ngrams.assign(ngrams=ngrams.ngram.apply(lambda x: [re.sub(r'_{2}', '-', w) for w in x]))

    # Separate words into different columns
    for i in range(1, max(nword) + 1):
        ngrams['word' + str(i)] = ngrams['ngrams'].apply(lambda x: x[i - 1] if len(x) >= i else '')

    ## come back to the original multiword format
    for i in range(1, max(nword) + 1):
        ngrams['word' + str(i)] = ngrams['word' + str(i)].str.replace('__', '-').str.replace('_', ' ')
    ##

    # Remove stopwords
    ngrams = ngrams.loc[~ngrams[['word' + str(i) for i in nword]].isin(stopwords).any(axis=1)]

    if stemming:
        ngrams = ngrams.assign(**{f'word{i}': lambda x: x[f'word{i}'].apply(lambda w: SnowballStemmer(language=language).stem(w)) for i in range(1, nword+1)})
    ngrams = ngrams.assign(ngram=lambda x: x[[f'word{i}' for i in range(1, nword+1)]].apply(lambda r: ' '.join(r)))
    ngrams = ngrams.assign(lambda x: x['ngram'].str.upper())
    # ngram = lambda x: x['ngram'].str.upper()
    ngrams = ngrams[~ngrams['ngram'].isin(custom_stopngrams)]
    if len(synonyms)>0 and isinstance(synonyms, str):
        s = [x.strip().upper() for x in synonyms.split(';')]
        snew = [x.split()[0] for x in s]
        sold = [[y for y in x.split()[1:] if y] for x in s]
        for i in range(len(s)):
            ngrams.loc[ngrams['ngram'].isin(sold[i]), 'ngram'] = snew[i]

    return ngrams