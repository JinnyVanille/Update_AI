# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import seaborn as sns

def authorProdOverTime(M, k=10, graph=True):
    if "DI" not in M.columns:
        M["DI"] = "NA"
    M["TC"] = pd.to_numeric(M["TC"])
    M["PY"] = pd.to_numeric(M["PY"])
    M = M.dropna(subset=["PY"])

    Y = int(str(datetime.now())[:4])
    M["AU"] = M["AU"].str.split(";")
    df = M.explode("AU")
    df = pd.DataFrame({"AU": df["AU"].str.strip(), "SR": df["SR"]})
    AU = df.groupby("AU").size().reset_index(name="n").sort_values(by="n", ascending=False).head(k)

    df = pd.merge(df, AU, on="AU", how="right")
    df = pd.merge(df, M, on="SR", how="left")
    df = df[["AU", "PY", "TI", "SO", "DI", "TC"]]
    df["TCpY"] = df["TC"] / (Y - df["PY"] + 1)
    df["n"] = df.groupby("AU")["AU"].transform("size")
    df = df.rename(columns={"AU": "Author", "PY": "year", "DI": "DOI"})
    df = df.sort_values(by=["n", "year"], ascending=[False, False])
    df = df.drop(columns=["n"])

    df2 = df.groupby(["Author", "year"]).agg({"TC": "sum", "TCpY": "sum", "year": "count"}).rename(
        columns={"year": "freq"})
    df2 = df2.reset_index()
    df2["Author"] = pd.Categorical(df2["Author"], categories=AU["AU"])

    x = [0.5, (1.5 * k) / 10]
    y = [df["year"].min(), df2["year"].min() + (df2["year"].max() - df2["year"].min()) * 0.125]

    logo = plt.imread("logo.png")
    imagebox = OffsetImage(logo, zoom=0.15)
    ab = AnnotationBbox(imagebox, xy=(x[0], y[0]), xycoords='data', frameon=False)

    g = sns.scatterplot(data=df2, x="Author", y="year", size="freq", alpha="TCpY", hue="TCpY", sizes=(2, 6),
                        palette="Blues_r")
    g.set(xlim=(AU["AU"].tolist()[::-1][0], AU["AU"].tolist()[::-1][-1]))
    g.set(ylim=(df2["year"].min(), df2["year"].max()))
    g.set(yticks=np.arange(df2["year"].min(), df2["year"].max(), step=2))
    g.set(xlabel="Author")
    g.set(ylabel="Year")
    g.set(title="Authors' Production over Time")
    g.add_artist(ab)
    g.legend(title="Legend", loc="center right", bbox_to_anchor=(1.3, 0.5), ncol=1, borderpad=1)

    if graph:
        plt.show()

    res = {"dfAU": df2, "dfPapersAU": df, "graph": g}

    return res
