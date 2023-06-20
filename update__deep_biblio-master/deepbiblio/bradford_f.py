# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import seaborn as sns

def bradford(M):
    SO = pd.Series(M['SO']).value_counts().sort_values(ascending=False)
    n = sum(SO)
    cumSO = np.cumsum(SO)
    cutpoints = [1, round(n*0.33), round(n*0.67), math.inf]
    groups = pd.cut(cumSO, bins=cutpoints, labels=["Zone 1", "Zone 2", "Zone 3"])
    a = len(np.where(cumSO < n*0.33)[0])+1
    b = len(np.where(cumSO < n*0.67)[0])+1
    Z = np.concatenate((np.repeat("Zone 1", a), np.repeat("Zone 2", b-a), np.repeat("Zone 3", len(cumSO)-b)))
    df = pd.DataFrame({'SO': SO.index, 'Rank': range(1, len(SO)+1), 'Freq': SO.values, 'cumFreq': cumSO, 'Zone': Z})

    SO = pd.Series(M['SO']).value_counts().sort_values(ascending=False)
    n = SO.sum()
    cumSO = SO.cumsum()
    cutpoints = np.round([1, n * 0.33, n * 0.67, np.inf])
    groups = pd.cut(cumSO, bins=cutpoints, labels=["Zone 1", "Zone 2", "Zone 3"])
    a = (cumSO < n * 0.33).sum() + 1
    b = (cumSO < n * 0.67).sum() + 1
    Z = pd.concat(
        [pd.Series(["Zone 1"] * a), pd.Series(["Zone 2"] * (b - a)), pd.Series(["Zone 3"] * (len(cumSO) - b))])
    df = pd.DataFrame(
        {"SO": cumSO.index, "Rank": np.arange(1, len(cumSO) + 1), "Freq": SO, "cumFreq": cumSO, "Zone": Z})

    x = [np.max(np.log(df['Rank'])) - 0.02 - np.diff(np.log(df['Rank']).min()) * 0.125,
         np.max(np.log(df['Rank'])) - 0.02]
    y = [df['Freq'].min(), df['Freq'].min() + np.diff(df['Freq']).min() * 0.125 + 1]
    logo = plt.imread('logo.png')
    img = OffsetImage(logo, zoom=0.15, interpolation="sinc")
    img.set_offset((x[0], y[0]))
    g = sns.lineplot(x=np.log(df['Rank']), y=df['Freq'], data=df, sort=False, linewidth=2)
    g.set(xscale="log", title="Core Sources by Bradford's Law", xlabel="Source log(Rank)", ylabel="Articles",
          ylim=(0, df['Freq'].max() + 10))
    g.annotate("Core\nSources", xy=(np.log(df['Rank'][a]) / 2, df['Freq'].max() / 2), fontsize=10, fontweight="bold",
               ha="center", va="center", alpha=0.5)
    g.axvline(x=np.log(df['Rank'][a]), ymin=0, ymax=df['Freq'].max(), color="gray", linestyle="--", alpha=0.2)
    g.set_xticks(np.log(df['Rank'][0:a]))
    g.set_xticklabels([x[:25] for x in df['SO'][0:a]], rotation=90, ha="center", fontsize=8, fontweight="bold")
    g.spines['right'].set_visible(False)
    g.spines['top'].set_visible(False)
    g.grid(axis="y", alpha=0.3)
    g.add_artist(img)

    results = {"table": df, "graph": g}
    return results