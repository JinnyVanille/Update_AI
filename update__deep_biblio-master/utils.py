# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import csv
from IPython.core.display import HTML
from igraph import plot as ig_plot
from igraph import VertexClustering  as centr_betw
from urllib.request import urlretrieve
from zipfile import ZipFile
import plotly.graph_objs as go
import plotly.express as px
import re
import numpy as np
import os
import pandas as pd
import urllib.request
import urllib.error

from deepbiblio.Hindex_f import Hindex
from deepbiblio.conceptualStructure_f import conceptualStructure
from deepbiblio.localCitations_f import localCitations
from deepbiblio.metaTagExtraction_f import metaTagExtraction
from deepbiblio.tableTag_f import tableTag


def getFileNameExtension(fn):
    # Split the file path using the OS-specific path separator
    splitted = fn.split(os.path.sep)
    # Get the file name from the splitted list
    fn = splitted[-1]
    ext = ''
    # Split the file name based on the '.' character
    splitted = fn.split('.')
    l = len(splitted)
    if l > 1 and any(splitted[0:(l-1)]):
        # Get the last item in the list as the extension
        ext = splitted[-1]
    # return the extension
    return ext

def strPreview(string, sep=","):
    # Split the string using the separator
    str1 = string.split(sep)
    # Get the first 5 items or all items if the length is less than 5
    str1 = str1[:min(len(str1), 5)]
    # Join the items back to a string using the separator
    str1 = sep.join(str1)
    # Display the preview as an HTML preformatted text
    return HTML(f"<pre>File Preview: {str1}</pre>")

def strSynPreview(string):
    # Get the first item in the list
    string = string[0]
    # Split the string using the semicolon separator
    str1 = string.split(";")
    # Get the first 5 items or all items if the length is less than 5
    str1 = str1[:min(len(str1), 5)]
    # Join the items back to a string
    str1 = " ".join([f"{item} <-" if idx==0 else item for idx, item in enumerate(str1)])
    # Display the preview as an HTML preformatted text
    return HTML(f"<pre>File Preview: {str1}</pre>")

def igraph2PNG(x, filename, width=10, height=7, dpi=75):
    # Add vertex attribute for centrality
    x.vs["centr"] = centr_betw(x)
    # Create dataframe with vertex label, color and centrality
    df = x.vs.attribute_values(["label", "color", "centr"], all=True)
    df = df.sort_values("centr", ascending=False)
    df = df.groupby("color").head(3)
    # Set vertex label to empty for vertices not in top 3 centrality for each cluster
    x.vs["label"] = ["" if label not in df["label"].tolist() else label for label in x.vs["label"]]
    # Plot the graph
    ig_plot(x, filename, bbox=(0, 0, width * dpi, height * dpi), vertex_label_size=12,
            vertex_size=20, vertex_frame_width=0.5, vertex_color=x.vs["color"], vertex_label=x.vs["label"])

def plot_ly(g, flip=False, side="r", aspectratio=1, size=0.15, data_type=2, height=0, customdata=None):
    # g = g + labs(title=None)

    gg = px.ggplotly(g, tooltip="text").update_layout(
        displaylogo=False,
        modebar_buttons_to_remove=[
            'toImage',
            'sendDataToCloud',
            'pan2d',
            'select2d',
            'lasso2d',
            'toggleSpikelines',
            'hoverClosestCartesian',
            'hoverCompareCartesian'
        ]
    )

    return gg



def freqPlot(xx, x, y, textLaby, textLabx, title, values):
    xl = [max(xx[x]) - 0.02 - (max(xx[x]) - min(xx[x])) * 0.125, max(xx[x]) - 0.02] + 1
    yl = [1, 1 + len(xx[y].unique()) * 0.125]

    Text = textLaby + ": " + xx[y].astype(str) + "\n" + textLabx + ": " + xx[x].astype(str)

    g = px.scatter(xx, x=xx[x], y=xx[y], text=Text, color=-xx[x], size=xx[x]).update_traces(
        textposition='top center',
        marker=dict(sizemode='radius', sizemin=5, sizemax=12, color=-xx[x], coloraxis='coloraxis')
    )

    g.update_layout(
        title=title,
        xaxis_title=textLabx,
        yaxis_title=textLaby,
        coloraxis=dict(colorscale='Viridis'),
        height=600,
        width=800,
        margin=dict(l=50, r=50, b=50, t=50, pad=4),
        paper_bgcolor="white",
        font=dict(family="Arial", size=12, color="#404040"),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        legend=dict(title="", orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial")
    )

    g.add_layout_image(
        dict(source=values['logoGrid'], xref="x", yref="y", x=xl[0], y=yl[0], xanchor="left", yanchor="bottom",
             sizex=xl[1] - xl[0], sizey=yl[1] - yl[0], sizing="stretch", opacity=1, layer="below")
    )

    return g

def emptyPlot(errortext):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        annotations=[dict(
            x=0.5,
            y=0.5,
            text=errortext,
            font=dict(size=20),
            showarrow=False,
            xref="paper",
            yref="paper"
        )]
    )
    fig.show()

def count_duplicates(df):
    x = pd.Series(df.apply(lambda row: '\r'.join(row.values.astype(str)), axis=1))
    ox = np.argsort(x)
    rl = np.diff(np.concatenate(([0], np.where(np.diff(x[ox]) != 0)[0]+1, [len(x)])))
    result = pd.concat([df.iloc[ox].reset_index(drop=True), pd.Series(rl, name='count')], axis=1)
    return result

def reduce_refs(A):
    pattern1 = re.compile(r'V[0-9]')
    pattern2 = re.compile(r'DOI')
    ind1 = pattern1.search(A).start() if pattern1.search(A) is not None else -1
    ind2 = pattern2.search(A).start() if pattern2.search(A) is not None else -1
    A = A[:ind1-1] if ind1 > -1 else A
    A = A[:ind2-1] if ind2 > -1 else A
    return A



def notifications():
    # Check internet connection and download notifications
    try:
        response = urllib.request.urlopen("https://www.bibliometrix.org/bs_notifications/biblioshiny_notifications.csv")
        notifOnline = pd.read_csv(response)
        notifOnline['href'] = notifOnline['href'].apply(lambda x: x if len(str(x)) >= 6 else None)
    except urllib.error.URLError:
        notifOnline = None

    # Check if a local file exists and load it
    home = os.path.expanduser("~")
    file_path = os.path.join(home, "biblioshiny_notifications.csv")
    fileTrue = os.path.exists(file_path)
    if fileTrue:
        notifLocal = pd.read_csv(file_path)

    # Determine status based on online/offline and local file present/absent
    A = ['noA', 'A']
    B = ['noB', 'B']
    status = A[int(not notifOnline)] + B[int(fileTrue)]

    # check connection and download notifications
    online = is_online()
    location = "https://www.bibliometrix.org/bs_notifications/biblioshiny_notifications.csv"
    notifOnline = None
    if online:
        # add check to avoid blocked app when internet connection is too slow
        envir = locals()
        # setTimeLimit(cpu=1, elapsed=1, transient=True)
        try:
            notifOnline = pd.read_csv(location, header=0, sep=",")
        except:
            notifOnline = None
        # setTimeLimit(cpu=Inf, elapsed=Inf, transient=False)
        if notifOnline is not None:
            notifOnline.loc[notifOnline['href'].str.len() < 6, 'href'] = pd.NA

    # check if a file exists on the local machine and load it
    home = os.path.expanduser("~")
    file = os.path.join(home, "biblioshiny_notifications.csv")
    fileTrue = os.path.exists(file)
    if fileTrue:
        notifLocal = pd.read_csv(file, header=0, sep=",")

    A = ["noA", "A"]
    B = ["noB", "B"]
    status = A[online] + B[fileTrue]

    if status == "noAnoB":
        notifTot = pd.DataFrame({'nots': ["No notifications"], 'href': [pd.NA], 'status': ["info"]})
    elif status == "noAB":
        notifTot = notifLocal.query('action == True').assign(status='info')
    elif status == "AnoB":
        notifOnline = notifOnline.iloc[:5]
        notifTot = notifOnline.query('action == True').assign(status='danger')
        notifOnline.query('action == True').to_csv(file, index=False, quoting=csv.QUOTE_NONE)
    elif status == "AB":
        notifTot = pd.merge(notifOnline.assign(status='danger'),
                            notifLocal.assign(status='info'), on='nots', how='left')
        notifTot['status'] = notifTot['status_y'].fillna("danger")
        notifTot.rename(columns={'href_x': 'href', 'action_x': 'action'}, inplace=True)
        notifTot = notifTot[['nots', 'href', 'action', 'status']]
        notifTot = notifTot.query('action == True').sort_values(by='status').iloc[:5]
        notifTot[['nots', 'href', 'action']].to_csv(file, index=False, quoting=csv.QUOTE_NONE)

    return status

import urllib.request

def is_online():
    try:
        urllib.request.urlopen('https://www.google.com')
        return True
    except:
        return False

def initial(values_para):
    values_para["results"] = ["NA"]
    values_para["log"] = "working..."
    values_para["load"] = "FALSE"
    values_para["field"] = values_para["cocngrams"] = "NA"
    values_para["citField"] = values_para["colField"] = values_para["citSep"] = "NA"
    values_para["NetWords"] = values_para["NetRefs"] = values_para["ColNetRefs"] = [[None]]
    values_para["Title"] = "Network"
    values_para["Histfield"] = "NA"
    values_para["histlog"] = "working..."
    values_para["kk"] = 0
    values_para["histsearch"] = "NA"
    values_para["citShortlabel"] = "NA"
    values_para["S"] = ["NA"]
    values_para["GR"] = "NA"
    return values_para




# Define a function to get the file extension
def get_file_extension(filename):
    return os.path.splitext(filename)[1][1:].lower()

# Define a function to convert files to dataframes
def convert_to_dataframe(file_path, dbsource, file_format):
    if file_format == "zip":
        with ZipFile(file_path) as zip_file:
            file_list = zip_file.namelist()
            # Assuming that there is only one file inside the zip file
            data_file_path = zip_file.extract(file_list[0])
            M = pd.read_csv(data_file_path)
    elif file_format == "csv":
        M = pd.read_csv(file_path)
    elif file_format == "bibtex":
        # Code to convert bibtex file to dataframe
        pass
    return M

def ValueBoxes(M):
    # calculate statistics for Biblioshiny ValueBoxes
    df = pd.DataFrame({'Description': [""] * 12, 'Results': [np.nan] * 12})

    ## VB  1 - Time span
    df.iloc[0, :] = ['Timespan', f"{np.nanmin(M['PY'])}:{np.nanmax(M['PY'])}"]

    ## VB  2 - Authors
    listAU = (M['AU'].str.split(";"))
    nAU = listAU.apply(len)
    listAU = np.unique([item.strip() for sublist in listAU for item in sublist if item])
    df.iloc[1, :] = ['Authors', len(listAU)]

    ## VB  3 - Author's Keywords (DE)
    DE = np.unique([item.strip() for sublist in M['DE'].str.split(";") for item in sublist if item])
    DE = DE[~pd.isnull(DE)]
    df.iloc[2, :] = ["Author's Keywords (DE)", len(DE)]

    ## VB  4 - Sources
    df.iloc[3, :] = ['Sources (Journals, Books, etc)', len(np.unique(M['SO']))]

    ## VB  5 - Authors of single-authored docs
    df.iloc[4, :] = ['Authors of single-authored docs', len(np.unique(M['AU'][nAU == 1]))]

    ## VB  6 - References
    CR = [item.strip() for sublist in M['CR'].str.split(";") for item in sublist if item]
    CR = [item for item in CR if len(item) > 0]
    df.iloc[5, :] = ['References', len(np.unique(CR))]

    ## VB  7 - Documents
    df.iloc[6, :] = ['Documents', len(M)]

    ## VB  8 - International Co-Authorship
    if 'AU_CO' not in M.columns:
        M = metaTagExtraction(M, "AU_CO")

    df = pd.DataFrame(columns=["Description", "Results"])

    # VB 1 - Time span
    df.loc[0] = ["Timespan", ":".join(map(str, [M["PY"].min(skipna=True), M["PY"].max(skipna=True)]))]

    # VB 2 - Authors
    listAU = M["AU"].str.split(";")
    nAU = listAU.apply(len)
    listAU = pd.Series(np.concatenate(listAU)).str.strip().dropna().unique()
    df.loc[1] = ["Authors", len(listAU)]

    # VB 3 - Author's Keywords (DE)
    DE = M["DE"].str.replace("\\s+|\\.|\\,", " ", regex=True).str.split(";").explode()
    DE = DE.str.strip().dropna().unique()
    df.loc[2] = ["Author's Keywords (DE)", len(DE)]

    # VB 4 - Sources
    df.loc[3] = ["Sources (Journals, Books, etc)", len(M["SO"].unique())]

    # VB 5 - Authors of single-authored docs
    df.loc[4] = ["Authors of single-authored docs", len(M["AU"][nAU == 1].unique())]

    # VB 6 - References
    CR = M["CR"].str.replace("\\s+|\\.|\\,", " ", regex=True).str.split(";").explode()
    CR = CR.str.strip().dropna().unique()
    df.loc[5] = ["References", len(CR)]

    # VB 7 - Documents
    df.loc[6] = ["Documents", len(M)]

    # VB 8 - International Co-Authorship
    if "AU_CO" not in M.columns:
        M = metaTagExtraction(M, "AU_CO")
    AU_CO = M["AU_CO"].str.split(";")
    Coll = pd.Series([len(set(l)) > 1 for l in AU_CO]).sum() / len(M) * 100
    df.loc[7] = ["International co-authorships %", "{:.4f}".format(Coll)]

    # VB 9 - Document Average Age
    age = pd.Timestamp.now().year - M["PY"].astype(int)
    df.loc[8] = ["Document Average Age", "{:.3f}".format(age.mean(skipna=True))]

    # VB 10 - Annual Growth Rate
    Y = M["PY"].value_counts().sort_index()
    ny = Y.index[-1] - Y.index[0]
    CAGR = round(((Y.iloc[-1] / Y.iloc[0]) ** (1 / ny) - 1) * 100, 2)
    df.loc[9] = ["Annual Growth Rate %", CAGR]

    # VB 11 - Co-Authors per Doc
    df.loc[10] = ["Co-Authors per Doc", "{:.3f}".format(nAU.mean(skipna=True))]

    # VB 12 - Average citations per doc
    df.loc[11] = ["Average citations per doc", "{:.4f}".format(M["TC"].mean(skipna=True))]

    # Create an empty data frame
    df = pd.DataFrame(columns=["Description", "Results"])

    # Indexed Keywords (ID)
    ID = M["ID"].str.split(";").explode().str.replace("\\s+|\\.|\\,", " ", regex=True).str.strip().unique()
    ID = ID[~pd.isna(ID)]
    df = df.append({"Description": "Keywords Plus (ID)", "Results": len(ID)}, ignore_index=True)

    # Single authored docs
    nAU = M["AU"].str.split(";").apply(len)
    df = df.append({"Description": "Single-authored docs", "Results": sum(nAU == 1)}, ignore_index=True)

    # Create a data frame for descriptions
    df2 = pd.DataFrame(
        {"Description": ["MAIN INFORMATION ABOUT DATA", "Timespan", "Sources (Journals, Books, etc)", "Documents",
                         "Annual Growth Rate %", "Document Average Age", "Average citations per doc", "References",
                         "DOCUMENT CONTENTS", "Keywords Plus (ID)", "Author's Keywords (DE)", "AUTHORS", "Authors",
                         "Authors of single-authored docs",
                         "AUTHORS COLLABORATION", "Single-authored docs", "Co-Authors per Doc",
                         "International co-authorships %", "DOCUMENT TYPES"]})

    # Join df2 and df, and replace NA values in Results column with empty string
    df = df2.merge(df, on="Description", how="left").append(
        M["DT"].str.lower().value_counts().rename_axis("Description").reset_index().rename(
            columns={"DT": "Results"})).fillna({"Results": ""})

    return df

def countryCollab(M):
    sep = ";"
    if "AU_CO" not in M.columns:
        M = metaTagExtraction(M, Field="AU_CO", sep=sep)
    if "AU1_CO" not in M.columns:
        M = metaTagExtraction(M, Field="AU1_CO", sep=sep)

    M["nCO"] = [len(set(x.split(sep))) > 1 for x in M["AU_CO"]]

    M["AU1_CO"] = M["AU1_CO"].str.strip().str.replace(r"\d+", "")
    M["AU1_CO"] = M["AU1_CO"].str.replace("UNITED STATES", "USA")
    M["AU1_CO"] = M["AU1_CO"].str.replace("RUSSIAN FEDERATION", "RUSSIA")
    M["AU1_CO"] = M["AU1_CO"].str.replace("TAIWAN", "CHINA")
    M["AU1_CO"] = M["AU1_CO"].str.replace("ENGLAND", "UNITED KINGDOM")
    M["AU1_CO"] = M["AU1_CO"].str.replace("SCOTLAND", "UNITED KINGDOM")
    M["AU1_CO"] = M["AU1_CO"].str.replace("WALES", "UNITED KINGDOM")
    M["AU1_CO"] = M["AU1_CO"].str.replace("NORTH IRELAND", "UNITED KINGDOM")

    df = (
        M.groupby("AU1_CO")
        .agg(
            Articles=("AU1_CO", "count"),
            SCP=("nCO", lambda x: sum(x == 0)),
            MCP=("nCO", lambda x: sum(x == 1)),
        )
        .reset_index()
        .rename(columns={"AU1_CO": "Country"})
        .sort_values(by=["Articles"], ascending=False)
    )

    return df


def hindex(values, type, input):
    if type == "author":
        AU = list(tableTag(values['M'], "AU"))
        AU = [name for name in AU if name != '']
        values['H'] = Hindex(values['M'], field="author", elements=AU, sep=";", years=float("inf"))['H']
        values['H'] = values['H'].sort_values(by=['h_index'], ascending=False)
    elif type == "source":
        SO = list(values['M']['SO'].value_counts().index)
        values['H'] = Hindex(values['M'], field="source", elements=SO, sep=";", years=float("inf"))['H']
        values['H'] = values['H'].sort_values(by=['h_index'], ascending=False)

    return values

def Hindex_plot(values, type, input):
    values = hindex(values, type=type, input=input)
    xx = values["H"]
    if type == "author":
        K = input["Hkauthor"]
        measure = input["HmeasureAuthors"]
        title = "Authors' Local Impact"
        xn = "Authors"
    else:
        K = input["Hksource"]
        measure = input["HmeasureSources"]
        title = "Sources' Local Impact"
        xn = "Sources"
    if K > xx.shape[0]:
        k = xx.shape[0]
    else:
        k = K

    if measure == "h":
        m = 1
    elif measure == "g":
        m = 2
    elif measure == "m":
        m = 3
        xx.iloc[:, m] = round(xx.iloc[:, m], 2)
    else:
        m = 4

    xx = xx.sort_values(by=m, ascending=False).iloc[:k, [0, m]]
    g = freqPlot(xx, x=2, y=1, textLaby=xn, textLabx=f"Impact Measure: {measure.upper()}",
                 title=f"{title} by {measure.upper()} index", values=values)

    res = {"values": values, "g": g}
    return res

def descriptive(values, type):
    if type == "tab2":
        TAB = (values["M"]
               .groupby('PY')
               .size()
               .reset_index(name='Articles')
               .rename(columns={'PY': 'Year'})
               .merge(pd.DataFrame({'Year': np.arange(values['M']['PY'].min(), values['M']['PY'].max() + 1)}),
                      how='right', on='Year')
               .fillna({'Articles': 0})
               .sort_values('Year')
               .reset_index(drop=True))

        ny = TAB['Year'].max() - TAB['Year'].min()
        values['GR'] = round(((TAB['Articles'].iloc[-1] / TAB['Articles'].iloc[0]) ** (1 / ny) - 1) * 100, digits=2)
    elif type == "tab3":
        listAU = list()
        index = list()
        num_x = 0
        for i in range(len(values["M"]["AU"])):
            tempv = values["M"]["AU"][i].split(";")
            for v in range(len(tempv)):
                listAU.append(tempv[v])
                index.append(num_x)
                num_x = num_x + 1
        nAU = len(listAU)
        fracAU = [1 / nAU] * nAU

        df = pd.DataFrame(columns=['Author', 'fracAU', 'index', 'AR'])
        df["Author"] = listAU
        df["fracAU"] = fracAU
        df["index"] = index
        df["AR"] = index
        # values["M"]["Author"] = listAU
        #v alues["M"]["fracAU"] = fracAU
        # values["M"]["index"] = index
        # TAB = pd.DataFrame({'Author': listAU, 'fracAU': fracAU})
        # TAB = (df.groupby('Author')
        #        .agg(Articles=('fracAU', 'size'), AuthorFrac=('fracAU', 'sum'))
        #        .reset_index()
        #        .sort_values('Articles', ascending=False)
        #        .rename(
        #     columns={'Author': 'Authors', 'Articles': 'Articles', 'AuthorFrac': 'Articles Fractionalized'})
        #        .reset_index(drop=True))
        TAB = df.groupby(df.loc[:, "Author"]).count().reset_index()
               # .agg(Articles=('fracAU', 'size'), AuthorFrac=('fracAU', 'sum'))
               # .reset_index()
               #.sort_values('Articles', ascending=False)
               #.rename(
            #columns={'Author': 'Authors', 'Articles': 'Articles', 'AuthorFrac': 'Articles Fractionalized'})
               #.reset_index(drop=True)
        TAB = TAB.rename(columns={'Author': 'Authors', 'AR': 'Articles', 'AuthorFrac': 'Articles Fractionalized'}) \
                .sort_values("Articles", ascending=False).reset_index(drop=True)

    elif type == "tab4":
        y = int(str(pd.Timestamp.now().date())[:4])
        TAB = (values["M"]
               .assign(TCperYear=lambda x: x['TC'] / (y + 1 - x['PY']))
               .loc[:, ['SR', 'DI', 'TC', 'TCperYear', 'PY']]
               .groupby('PY')
               .apply(lambda x: x.assign(NTC=x['TC'] / x['TC'].mean()))
               .reset_index(drop=True)
               .drop('PY', axis=1)
               .sort_values('TC', ascending=False)
               .reset_index(drop=True))
        TAB.columns = ['Paper', 'DOI', 'Total Citations', 'TC per Year', 'Normalized TC']
    elif type == "tab5":
        TAB = countryCollab(values['M'])
        TAB = (TAB.assign(Freq=TAB['Articles'] / TAB['Articles'].sum(),
                          MCP_Ratio=TAB['MCP'] / TAB['Articles'])
               .reset_index(drop=True))
    elif type == "tab6":
        if "AU1_CO" not in values["M"].columns:
            values["M"] = metaTagExtraction(values["M"], "AU1_CO")
        TAB = (values["M"]
               .loc[:, ["AU1_CO", "TC"]]
               .dropna(subset=["AU1_CO"])
               .rename(columns={"AU1_CO": "Country", "TC": "TotalCitation"})
               .groupby("Country")
               .agg(TC=("TotalCitation", "sum"),
                    Average_Article_Citations=("TotalCitation", lambda x: round(x.mean(), 1)))
               .reset_index()
               .sort_values(by=["TC"], ascending=False))
    elif type == "tab7":
        # values["Source title"] = np.array(values["M"].values[:][:])[:, 4]
        TAB = values["M"].groupby(values["M"].loc[:, "SO"]).count().reset_index()
        TAB = TAB.rename(columns={"SO": "Source title", "AB": "Number of document"})\
            .sort_values("TI", ascending=False).reset_index(drop=True)
    elif type == "tab10":
        TAB = mapworld(values["M"])["tab"]
    elif type == "tab11":
        if "AU_UN" not in values["M"].columns:
            values["M"] = metaTagExtraction(values["M"], Field="AU_UN")
        TAB = (pd.DataFrame({"Affiliation": values["M"]["AU_UN"].str.split(";").sum()})
               .groupby("Affiliation")
               .size()
               .reset_index(name="Articles")
               .dropna(subset=["Affiliation"])
               .sort_values(by=["Articles"], ascending=False)
               .reset_index(drop=True))
    elif type == "tab12":
        TAB = tableTag(values["M"], "C1")
        TAB = (pd.DataFrame({"Affiliations": list(TAB.keys()), "Articles": list(TAB.values())})
               .query("Affiliations.str.len() > 4")
               .reset_index(drop=True))
    elif type == "tab13":
        CR = localCitations(values["M"], fast_search=False, verbose=False)
        TAB = CR["Authors"]
        # TAB = pd.DataFrame({"Authors": list(CR["Authors"]["Author"]), "Citations": list(CR["Cited"])})
    else:
        raise ValueError("Invalid type provided")

    values["TAB"] = TAB
    res = {"values": values, "TAB": TAB}
    return res


def AffiliationOverTime(values, n):
    if "AU_UN" not in values['M'].columns:
        values['M'] = metaTagExtraction(values['M'], Field="AU_UN")
    AFF = values['M']['AU_UN'].str.split(";")
    nAFF = [len(x) for x in AFF]

    AFFY = pd.DataFrame({'Affiliation': [item for sublist in AFF for item in sublist],
                         'Year': np.repeat(values['M']['PY'], nAFF)}) \
            .dropna(subset=['Affiliation', 'Year']) \
            .groupby(['Affiliation', 'Year']) \
            .size() \
            .reset_index(name='Articles') \
            .groupby(['Affiliation']) \
            .apply(lambda x: x.sort_values(['Year'])) \
            .reset_index(drop=True) \
            .pivot(index='Affiliation', columns='Year', values='Articles') \
            .fillna(0) \
            .reset_index() \
            .melt(id_vars=['Affiliation'], var_name='Year', value_name='Articles') \
            .groupby(['Affiliation']) \
            .apply(lambda x: x.assign(Articles=x['Articles'].cumsum())) \
            .reset_index(drop=True)

    Affselected = AFFY \
        .query('Year == @AFFY.Year.max()') \
        .nlargest(n, 'Articles')

    values['AffOverTime'] = AFFY \
        .query('Affiliation in @Affselected.Affiliation') \
        .assign(Year=lambda x: x['Year'].astype(int))

    return values


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import seaborn as sns


def AffiliationOverTime(values, n):
    Text = [f"{a} ({y}) {n}" for a, y, n in
            zip(values['AffOverTime']['Affiliation'], values['AffOverTime']['Year'], values['AffOverTime']['Articles'])]
    width_scale = 1.7 * 26 / len(np.unique(values['AffOverTime']['Affiliation']))
    x = [max(values['AffOverTime']['Year']) - 0.02 - np.diff(np.array(values['AffOverTime']['Year'])) * 0.15,
         max(values['AffOverTime']['Year']) - 0.02 + 1]
    y = [min(values['AffOverTime']['Articles']),
         min(values['AffOverTime']['Articles']) + np.diff(np.array(values['AffOverTime']['Articles'])) * 0.15]

    # create plot
    logo = mpimg.imread(values['logoPath'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Year', y='Articles', data=values['AffOverTime'], hue='Affiliation', style='Affiliation', ax=ax)
    plt.title("Affiliations' Production over Time")
    plt.xlabel('Year')
    plt.ylabel('Articles')
    ax.xaxis.set_ticks(np.arange(min(values['AffOverTime']['Year']), max(values['AffOverTime']['Year']) + 1,
                                 int(np.ceil(len(values['AffOverTime']['Year']) / 20))))
    plt.axhline(y=0, alpha=0.1)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=5)
    plt.text(0.99, 0.01, values['caption'], fontsize=9, transform=plt.gcf().transFigure, ha='right', va='bottom')
    plt.imshow(logo, extent=[x[0], x[1], y[0], y[1]], aspect='auto', alpha=0.6)
    plt.axis('tight')
    plt.tight_layout()

    # add text to plot
    for i, t in enumerate(Text):
        offsetbox = TextArea(t, minimumdescent=False)
        xy = (values['AffOverTime']['Year'][i], values['AffOverTime']['Articles'][i])
        ab = AnnotationBbox(offsetbox, xy,
                            xybox=(0, 0),
                            xycoords='data',
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)

    values['AffOverTimePlot'] = fig

    Text = [f"{aff} ({year}) {art}" for aff, year, art in zip(values['AffOverTime']['Affiliation'],
                                                              values['AffOverTime']['Year'],
                                                              values['AffOverTime']['Articles'])]
    width_scale = 1.7 * 26 / len(np.unique(values['AffOverTime']['Affiliation']))
    x = [max(values['AffOverTime']['Year'])-0.02-np.diff(np.array(values['AffOverTime']['Year']))*0.15,
         max(values['AffOverTime']['Year'])-0.02+1]
    y = [min(values['AffOverTime']['Articles']),
         min(values['AffOverTime']['Articles'])+np.diff(np.array(values['AffOverTime']['Articles']))*0.15]

    fig, ax = plt.subplots()
    ax.plot(values['AffOverTime']['Year'], values['AffOverTime']['Articles'],
            linewidth=1.5, linestyle='-', alpha=0.7, color='blue', label='Affiliation', zorder=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Articles')
    ax.set_title("Affiliations' Production over Time")

    x_ticks = np.linspace(values['AffOverTime']['Year'][0], values['AffOverTime']['Year'][-1], 20)
    ax.set_xticks(x_ticks)

    ax.axhline(y=0, linewidth=0.5, color='gray', alpha=0.1, zorder=1)

    ax.legend(title='Affiliation', loc='lower center', fontsize=width_scale,
              title_fontsize=1.5*width_scale, ncol=1, bbox_to_anchor=(0.5, -0.35),
              fancybox=True, shadow=True)

    plt.setp(ax.get_xticklabels(), fontsize=10, rotation=90)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    plt.subplots_adjust(left=0.15, bottom=0.15)

    ax.annotate(values['logoGrid'], xy=(x[1], y[1]), xytext=(x[1], y[1]))

    plt.show()
    return values


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_line, labs, scale_x_continuous, geom_hline, \
    theme, element_text, element_rect, element_line#,    annotation_custom

#from grid import unit
from matplotlib import pyplot as plt

def CountryOverTime(values, n):
    if 'AU_CO' not in values['M'].keys():
        values['M'] = metaTagExtraction(values['M'], Field="AU_CO")

    AFF = [x.split(";") for x in values['M']['AU_CO']]
    nAFF = [len(x) for x in AFF]

    AFFY = pd.DataFrame(
        {'Affiliation': np.repeat(np.concatenate(AFF), nAFF), 'Year': np.repeat(values['M']['PY'], nAFF)})
    AFFY.dropna(subset=['Affiliation', 'Year'], inplace=True)
    AFFY = AFFY.groupby(['Affiliation', 'Year']).size().reset_index(name='Articles')
    AFFY = AFFY.groupby('Affiliation').apply(lambda x: x.sort_values('Year')).reset_index(drop=True)
    AFFY = pd.pivot_table(AFFY, index='Affiliation', columns='Year', values='Articles', aggfunc=np.sum, fill_value=0)
    AFFY = AFFY.apply(lambda x: x.cumsum(), axis=1)
    AFFY = AFFY.stack().reset_index(name='Articles').rename(columns={'level_1': 'Year'})
    AFFY['Year'] = AFFY['Year'].astype(int)

    Affselected = AFFY[AFFY['Year'] == AFFY['Year'].max()].sort_values(by='Articles', ascending=False).head(n)

    values['CountryOverTime'] = AFFY[AFFY['Affiliation'].isin(Affselected['Affiliation'])].rename(
        columns={'Affiliation': 'Country'})

    Text = values['CountryOverTime'].apply(lambda row: f"{row['Country']} ({row['Year']}) {row['Articles']}", axis=1)
    width_scale = 1.7 * 26 / len(values['CountryOverTime']['Country'].unique())
    x = [values['CountryOverTime']['Year'].max() - 0.02 - (
                values['CountryOverTime']['Year'].max() - values['CountryOverTime']['Year'].min()) * 0.15,
         values['CountryOverTime']['Year'].max() - 0.02 + 1]
    y = [values['CountryOverTime']['Articles'].min(), values['CountryOverTime']['Articles'].min() + (
                values['CountryOverTime']['Articles'].max() - values['CountryOverTime']['Articles'].min()) * 0.15]

    width_scale = 1  # set the width scale

    CountryOverTimePlot = ggplot(values.CountryOverTime,
                                 aes(x='Year', y='Articles', group='Country', color='Country', text='Text')) + \
                          geom_line() + \
                          labs(x='Year', y='Articles', title='Country Production over Time') + \
                          scale_x_continuous(
                              breaks=values.CountryOverTime.Year[0::int(len(values.CountryOverTime.Year) / 20)]) + \
                          geom_hline(aes(yintercept=0), alpha=0.1) + \
                          labs(color='Country') + \
                          theme(text=element_text(color='#444444'),
                                legend_text=element_text(size=width_scale),
                                legend_box_margin=6,
                                legend_title=element_text(size=1.5 * width_scale, face="bold"),
                                legend_position="bottom",
                                legend_direction="vertical",
                                # legend_key_size=unit(width_scale / 50, "inch"),
                                # legend_key_width=unit(width_scale / 50, "inch"),
                                legend_key_size=width_scale / 50,
                                legend_key_width=width_scale / 50,
                                plot_caption=element_text(size=9, hjust=0.5, color="black", face="bold"),
                                panel_background=element_rect(fill='#FFFFFF'),
                                panel_grid_minor=element_line(color='#EFEFEF'),
                                panel_grid_major=element_line(color='#EFEFEF'),
                                plot_title=element_text(size=24),
                                axis_title=element_text(size=14, color='#555555'),
                                axis_title_y=element_text(vjust=1, angle=90),
                                axis_title_x=element_text(hjust=0.95, angle=0),
                                axis_text_x=element_text(size=10, angle=90),
                                axis_line_x=element_line(color="black", size=0.5),
                                axis_line_y=element_line(color="black", size=0.5)),# + \
                  #        annotation_custom(values.logoGrid, xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1])
    values.logoGrid, xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1]

    values['CountryOverTimePlot'] = CountryOverTimePlot

    return values

def wordlist(M, Field, n, measure, ngrams=None, remove_terms=None, synonyms=None):
    if Field == "ID":
        v = tableTag(M, "ID", remove_terms=remove_terms, synonyms=synonyms)
    elif Field == "DE":
        v = tableTag(M, "DE", remove_terms=remove_terms, synonyms=synonyms)
    elif Field == "TI":
        if "TI_TM" not in M.columns:
            v = tableTag(M, "TI", ngrams=ngrams, remove_terms=remove_terms, synonyms=synonyms)
    elif Field == "AB":
        if "AB_TM" not in M.columns:
            v = tableTag(M, "AB", ngrams=ngrams, remove_terms=remove_terms, synonyms=synonyms)
    elif Field == "WC":
        v = tableTag(M, "WC")
    v.index = v.index.str.lower()
    n = min(n, len(v))
    Words = pd.DataFrame({"Terms": v.index[:n], "Frequency": v.values[:n]})
    W = Words.copy()
    if measure == "sqrt":
        W["Frequency"] = np.sqrt(W["Frequency"])
    elif measure == "log":
        W["Frequency"] = np.log(W["Frequency"] + 1)
    elif measure == "log10":
        W["Frequency"] = np.log10(W["Frequency"] + 1)

    results = {"v": v, "W": W, "Words": Words}
    return results

def readStopwordsFile(file, sep=","):
    if file is not None:
        data = pd.read_csv(file)
        remove_terms = data.iloc[:, 0].values.tolist()
    else:
        remove_terms = None
    return remove_terms

def readSynWordsFile(file, sep=","):
    if file is not None:
        data = pd.read_csv(file)
        syn_terms = data.iloc[:, 0].values.tolist()
        if sep != ";":
            syn_terms = [term.replace(sep, ";") for term in syn_terms]
    else:
        syn_terms = None
    return syn_terms


import pandas as pd
import numpy as np
import re
import plotly.express as px


def mapworld(M, values):
    if "AU_CO" not in M.columns:
        M = metaTagExtraction(M, "AU_CO")
    CO = pd.DataFrame(tableTag(M, "AU_CO"))
    CO.columns = ["Tab", "Freq"]
    CO["Tab"] = CO["Tab"].str.replace(r"\d+", "")
    CO["Tab"] = CO["Tab"].str.replace(".", "")
    CO["Tab"] = CO["Tab"].str.replace(";;", ";")
    CO["Tab"] = CO["Tab"].str.replace("UNITED STATES", "USA")
    CO["Tab"] = CO["Tab"].str.replace("RUSSIAN FEDERATION", "RUSSIA")
    CO["Tab"] = CO["Tab"].str.replace("TAIWAN", "CHINA")
    CO["Tab"] = CO["Tab"].str.replace("ENGLAND", "UNITED KINGDOM")
    CO["Tab"] = CO["Tab"].str.replace("SCOTLAND", "UNITED KINGDOM")
    CO["Tab"] = CO["Tab"].str.replace("WALES", "UNITED KINGDOM")
    CO["Tab"] = CO["Tab"].str.replace("NORTH IRELAND", "UNITED KINGDOM")
    CO["Tab"] = CO["Tab"].str.replace("UNITED KINGDOM", "UK")
    CO["Tab"] = CO["Tab"].str.replace("KOREA", "SOUTH KOREA")

    map_world = px.data.gapminder()
    map_world["region"] = map_world["iso_alpha"].str.upper()

    country_prod = pd.merge(map_world, CO, how="left", left_on="region", right_on="Tab")

    tab = country_prod.groupby("region")["Freq"].mean().reset_index()

    tab = tab[~np.isnan(tab['Freq'])]
    tab = tab.sort_values(by='Freq', ascending=False)

    breaks = pd.cut(CO['Freq'], bins=10, labels=False)
    breaks = dict(zip(breaks, breaks))

    g = px.choropleth(country_prod,
                      locations='.data$region',
                      locationmode='country names',
                      color='.data$Freq',
                      hover_data=[('Country', '.data$region'), ('N.Documents', '.data$Freq')],
                      range_color=[0, CO['Freq'].max()],
                      color_continuous_scale=[(0, '#87CEEB'), (1, 'dodgerblue4')],
                      labels={'N.Documents': 'N.Documents'},
                      title='Country Scientific Production',
                      width=None, height=None)

    g.update_layout(legend=dict(title=dict(text='N.Documents', font=dict(size=14))),
                    font=dict(color='#333333'),
                    xaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    legend_title_font=dict(size=14),
                    legend_title_text='',
                    # legend=dict(bgcolor='rgba(0,0,0,0)'),
                    annotations=[dict(showarrow=False, x=0.75, y=0.5,
                                      xref='paper', yref='paper',
                                      text=values['logoGrid'],
                                      font=dict(size=14),
                                      xanchor='left', yanchor='middle',
                                      opacity=1)],
                    margin=dict(l=0, r=0, t=80, b=0)
                    )

    results = {'g': g, 'tab': tab}

    return results

def CAmap(input, values):
    if input["CSfield"] in values["M"].columns:
        if input["CSfield"] in ["TI", "AB"]:
            ngrams = int(input["CSngrams"])
        else:
            ngrams = 1

        # load file with terms to remove
        if input["CSStopFile"] == "Y":
            remove_terms = (readStopwordsFile(file=input["CSStop"], sep=input["CSSep"])).strip()
        else:
            remove_terms = None
        values["CSremove.terms"] = remove_terms
        # end of block

        # load file with synonyms
        if input["FASynFile"] == "Y":
            synonyms = (readSynWordsFile(file=input["FASyn"], sep=input["FASynSep"])).strip()
        else:
            synonyms = None
        values["FAsyn.terms"] = synonyms
        # end of block
        if input["CSfield"] in values["M"].columns:
            if input["CSfield"] in ["TI", "AB"]:
                ngrams = int(input["CSngrams"])
            else:
                ngrams = 1

            if input["CSStopFile"] == "Y":
                remove_terms = readStopwordsFile(file=input["CSStop"], sep=input["CSSep"]).apply(str.strip)
            else:
                remove_terms = None

            if input["FASynFile"] == "Y":
                synonyms = readSynWordsFile(file=input["FASyn"], sep=input["FASynSep"]).apply(str.strip)
            else:
                synonyms = None

            tab = tableTag(values["M"], input["CSfield"], ngrams=ngrams)
            if len(tab) >= 2:
                min_degree = int(tab[input["CSn"]])
                cs = conceptualStructure(
                    values["M"],
                    method=input["method"],
                    field=input["CSfield"],
                    minDegree=min_degree,
                    clust=input["nClustersCS"],
                    k_max = 8,
                            stemming = False,
                                       labelsize = input["CSlabelsize"] / 2,
                                                   documents = input["CSdoc"],
                                                               graph = False,
                                                                       ngrams = ngrams,
                                                                                remove_terms = remove_terms,
                                                                                               synonyms = synonyms,
                )
                if input["method"] != "MDS":
                    cs_data = cs["docCoord"]
                    cs_data = pd.DataFrame(
                        {
                            "Documents": cs_data.index,
                            "dim1": round(cs_data["dim1"], 2),
                            "dim2": round(cs_data["dim2"], 2),
                            "contrib": round(cs_data["contrib"], 2),
                        }
                    )
                    cs["CSData"] = cs_data
                else:
                    cs["CSData"] = pd.DataFrame({"Documents": None, "dim1": None, "dim2": None})

                if input["method"] == "CA":
                    w_data = pd.DataFrame(
                        {
                            "word": cs["km.res"]["data.clust"].index,
                            **cs["km.res"]["data.clust"].to_dict(orient="series"),
                        }
                    )
                    w_data.rename(columns={"cluster": "Dim.1"}, inplace=True)
                elif input["method"] == "MCA":
                    w_data = pd.DataFrame(
                        {
                            "word": cs["km.res"]["data.clust"].index,
                            **cs["km.res"]["data.clust"].to_dict(orient="series"),
                        }
                    )
                    w_data.rename(columns={"cluster": "Dim.1"}, inplace=True)
                else:  # input["method"] == "MDS"
                    w_data = pd.DataFrame(
                        {
                            "word": cs["res"].index,
                            **cs["res"].to_dict(orient="series"),
                            "cluster": cs["km.res"]["cluster"],
                        }
                    )

                w_data["Dim.1"] = round(w_data["Dim.1"], 2)
                w_data["Dim.2"] = round(w_data["Dim.2"], 2)
                cs["WData"] = w_data
            else:
                emptyPlot("Selected field is not included in your data collection")
                cs = {"NA"}
        else:
            emptyPlot("Selected field is not included in your data collection")
            cs = {"NA"}

        values["CS"] = cs