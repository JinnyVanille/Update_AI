# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd
import numpy as np
from deepbiblio.AbbrevTitle_f import AbbrevTitle

def csvScopus2df(file):
    ## import all files in a single data frame
    # for i in range(len(file)):
    #     D = pd.read_csv(file[i], na_values='', quotechar='"', skipinitialspace=True, dtype=str)
    #     if i > 0:
    #         l = list(set(l).intersection(set(D.columns)))
    #         DATA = pd.concat([DATA[l], D[l]])
    #     else:
    #         l = list(D.columns)
    #         DATA = D.copy()

    D = pd.read_csv(file, na_values='', quotechar='"', skipinitialspace=True, dtype=str)

    l = list(D.columns)
    DATA = D.copy()

    ## Post-Processing

    # column re-labelling
    DATA = labelling(DATA)

    # Authors' names cleaning (surname and initials)
    DATA['AU'] = DATA['AU'].str.replace('\\.', '')
    DATA['AU'] = DATA['AU'].str.replace(',', ';')

    # Affiliation
    if 'C1' not in DATA.columns:
        DATA['C1'] = np.nan
    else:
        DATA['C1'] = DATA['C1'].str.split(';').apply(lambda l: ';'.join([x.split(', ')[0] for x in l]))

    # Iso Source Titles
    if 'JI' in DATA.columns:
        DATA['J9'] = DATA['JI'].str.replace('\\.', '')
    else:
        # DATA['JI'] = DATA['SO'].apply(AbbrevTitle, USE.NAMES=False)
        DATA['JI'] = DATA['SO'].apply(lambda x: AbbrevTitle(x) if isinstance(x, str) else None)
        DATA['J9'] = DATA['JI'].str.replace('\\.', '')

    DI = DATA['DI'].copy()
    URL = DATA['URL'].copy()
    DATA = DATA.apply(lambda x: x.str.upper() if isinstance(x[0], str) else x)
    DATA['DI'] = DI
    DATA['URL'] = URL
    return DATA

def labelling(DATA):
  ## column re-labelling

  df_tag = pd.DataFrame(
      [
          ["Abbreviated Source Title","JI"],
          ["Authors with affiliations","C1"],
          ["Author Addresses","C1"],
          ["Authors","AU"],
          ["Author Names","AU"],
          ["Author full names", "AF"],
          ["Source title","SO"],
          ["Titles","TI"],
          ["Title","TI"],
          ["Publication Year","PY"],
          ["Year","PY"],
          ["Volume","VL"],
          ["Issue","IS"],
          ["Page count","PP"],
          ["Cited by","TC"],
          ["DOI","DI"],
          ["Link","URL"],
          ["Abstract","AB"],
          ["Author Keywords","DE"],
          ["Indexed Keywords","ID"],
          ["Index Keywords","ID"],
          ["Funding Details","FU"],
          ["Funding Texts","FX"],
          ["Funding Text 1","FX"],
          ["References","CR"],
          ["Correspondence Address","RP"],
          ["Publisher","PU"],
          ["Open Access","OA"],
          ["Language of Original Document","LA"],
          ["Document Type","DT"],
          ["Source","DB"],
          ["EID","UT"]
      ],
      columns=["orig", "tag"]
  )

  label = pd.DataFrame({"orig": DATA.columns}).merge(
      df_tag, how="left", on="orig"
  ).assign(
      tag=lambda x: np.where(pd.isna(x["tag"]), x["orig"], x["tag"])
  )

  DATA.columns = label["tag"]

  return DATA