# # author: Tang Tiong Yew
# # email: tiongyewt@sunway.edu.my
# # Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# # Copyright 2023
# #
import numpy as np
import pandas as pd
import prince
from prince import CA, MCA
from sklearn import manifold
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize
import plotly.figure_factory as ff

from deepbiblio.cocMatrix_f import cocMatrix
from deepbiblio.colSums_f import colSums
from deepbiblio.colnames_f import colnames
from deepbiblio.rowSums_f import rowSums
from deepbiblio.termExtraction_f import termExtraction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import seaborn as sns

def conceptualStructure(M, field="ID", ngrams=1, method="MCA", quali_supp=None, quanti_supp=None, minDegree=2,
                        clust="auto", k_max=5, stemming=False, labelsize=10, documents=2, graph=True, remove_terms=None,
                        synonyms=None):
    cbPalette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    # if quali_supp is not None:
    #     QSUPP = M[quali_supp]
    #     QSUPP.columns = M.columns[quali_supp]
    #     QSUPP.index = M.index.str.lower()
    #
    # if quanti_supp is not None:
    #     SUPP = M[quanti_supp]
    #     SUPP.columns = M.columns[quanti_supp]
    #     SUPP.index = M.index.str.lower()
    #
    # binary = False
    # if method == "MCA":
    #     binary = True
    #
    # if field == "ID":
    #     # Create a bipartite network of Keyword plus
    #     #
    #     # each row represents a manuscript
    #     # each column represents a keyword (1 if present, 0 if absent in a document)
    #     CW = cocMatrix(M, Field="ID", type="matrix", sep=";", binary=binary, remove_terms=remove_terms,
    #                    synonyms=synonyms)
    #     # Define minimum degree (number of occurrences of each Keyword)
    #     CW = CW[:, colSums(CW) >= minDegree]
    #     # Delete empty rows
    #     CW = CW[:, ~(colnames(CW) == "NA")]
    #     CW = CW[rowSums(CW) > 0]
    #
    # elif field == "TI":
    #     M = termExtraction(M, Field="TI", remove_numbers=True, stemming=stemming, language="english",
    #                        remove_terms=remove_terms, synonyms=synonyms, keep_terms=None, verbose=False, ngrams=ngrams)
    #
    #     CW = cocMatrix(M, Field="TI_TM", type="matrix", sep=";", binary=binary)
    #     # Define minimum degree (number of occurrences of each Keyword)
    #     CW = CW[:, colSums(CW) >= minDegree]
    #     # Delete empty rows
    #     CW = CW[:, ~(colnames(CW) == "NA")]
    #     CW = CW[rowSums(CW) > 0,]
    #
    # elif field == "AB":
    #     M = termExtraction(M, Field="AB", remove_numbers=True, stemming=stemming, language="english",
    #                        remove_terms=remove_terms, synonyms=synonyms, keep_terms=None, verbose=False, ngrams=ngrams)
    #
    #     CW = cocMatrix(M, Field="AB_TM", type="matrix", sep=";", binary=binary)
    #     # Define minimum degree (number of occurrences of each Keyword)
    #     CW = CW[:, colSums(CW) >= minDegree]
    #     # Delete empty rows
    #     CW = CW[rowSums(CW) > 0,]
    #     CW = CW[:, ~(colnames(CW) == "NA")]
    #     # Recode as dataframe
    #     # CW = pd.DataFrame(CW.apply(lambda col: pd.Categorical(col)))
    #
    # km_res_centers = centers[:, [1, 2, 0]]
    #
    # # It seems like logo is a data object or a file name that needs to be loaded
    # # into the environment before the function is called. You can load it using
    # # a Python library like Pillow or OpenCV.
    # logo = ...
    #
    # df_clust = pd.DataFrame(km_res_data_clust)
    # df_clust["shape"] = "1"
    # df_clust["label"] = df_clust.index
    # centers_df = pd.DataFrame(km_res_centers, columns=["Dim.1", "Dim.2", ".cluster"])
    # centers_df["shape"] = "0"
    # centers_df["label"] = ""
    # df_clust = pd.concat([df_clust, centers_df])
    # df_clust["color"] = colorlist()[df_clust[".cluster"]]
    #
    # hull_data = df_clust.groupby(".cluster").apply(
    #     lambda x: x.iloc[x[["Dim.1", "Dim.2"]].values.astype(float).convex_hull])
    # hull_data = pd.concat([hull_data, hull_data.groupby("clust").head(1)])
    # hull_data = hull_data.reset_index(drop=True)
    # hull_data["id"] = np.arange(len(hull_data))
    # hull_data = hull_data.sort_values(by=["clust", "id"]).reset_index(drop=True)
    #
    # size = labelsize
    # centers = centers[:, [1, 2, 0]]
    #
    # logo = importr("grid").rasterGrob(logo, interpolate=True)
    #
    # df_clust = km_res["data.clust"].assign(shape="1", label=km_res["data.clust"].index).append(
    #     centers.assign(shape="0", label="")
    # ).assign(color=colorlist()[km_res["data.clust"]])
    #
    # hull_data = df_clust.groupby(km_res["data.clust"]).apply(
    #     lambda x: x.iloc[chull(x["Dim.1"], x["Dim.2"])]
    # ).reset_index(drop=True)
    #
    # hull_data = hull_data.append(hull_data.groupby("clust").head(1)).assign(id=range(1, len(hull_data) + 1)).sort_values(["clust", "id"])
    #
    # size = labelsize
    #
    # b = ggplot2.ggplot(df_clust) + \
    #     ggplot2.aes_string(x=".data$Dim.1", y=".data$Dim.2", shape=".data$shape", color=".data$color") + \
    #     ggplot2.geom_point() + \
    #     ggplot2.geom_polygon(data=hull_data, mapping=ggplot2.aes_string(fill=".data$color", colour=".data$color"),
    #                          alpha=0.3, show_legend=False) + \
    #     ggrepel.geom_text_repel(ggplot2.aes_string(label=".data$label")) + \
    #     ggplot2.theme_minimal() + \
    #     ggplot2.labs(title=paste("Conceptual Structure Map - method: ", method, collapse="", sep="")) + \
    #     ggplot2.geom_hline(yintercept=0, linetype="dashed", color=utils.adjustcolor("grey40", alpha_f=0.7)) + \
    #     ggplot2.geom_vline(xintercept=0, linetype="dashed", color=utils.adjustcolor("grey40", alpha_f=0.7)) + \
    #     ggplot2.theme(text=ggplot2.element_text(size=size),
    #                   axis_title=ggplot2.element_text(size=size, face="bold"),
    #                   plot_title=ggplot2.element_text(size=size + 1, face="bold"),
    #                   panel_background=ggplot2.element_rect(fill="white", colour="white"),
    #                   axis_line_x=ggplot2.element_line(color="black", linewidth=0.5),
    #                   axis_line_y=ggplot2.element_line(color="black", linewidth=0.5),
    #                   panel_grid_major=ggplot2.element_blank(),
    #                   panel_grid_minor=ggplot2.element_blank())
    #
    # if method != "MDS":
    #     b = b + ggplot2.xlab(paste("Dim 1 (", round(res_mca.eigCorr.perc[0] * 100, 2), "%)", sep="")) + \
    #         ggplot2.ylab(paste("Dim 2 (", round(res_mca.eigCorr.perc[1] * 100, 2), "%)", sep=""))
    # else:
    #     b = b + ggplot2.xlab("Dim 1") + ggplot2.ylab("Dim 2")
    #
    #

def dendPlot(km_res, clust, label_cex, graph=False):
    # # Dendrogram object
    # dend < - as.dendrogram(km_res)
    # # vector of colors
    # labelColors = colorlist()[0:clust - 1]
    #
    # # cut dendrogram in k clusters
    # clusMember = cutree(km_res, clust)
    #
    # # function to get color labels
    # def colLab(n):
    #     if (is.leaf(n)):
    #         a = attributes(n)
    #         labCol = labelColors[clusMember[names(clusMember) == a$label][0]]
    #         attr(n, "nodePar") = [a$nodePar$lab.col, {"lab.col": labCol, "lab.cex": label_cex}]
    #         # attr(n, "label_cex") <- c(a$nodePar$lab.cex, label_cex = 0.1)
    #         return n
    #
    #     # using dendrapply
    #     clusDendro < - dendrapply(dend, colLab)
    #     k = clust
    #     n = len(km_res["labels"])
    #     MidPoint = (km_res["height"][n - k] + km_res["height"][n - k + 1]) / 2
    #
    #     plotRes = {"dend": clusDendro, "line": MidPoint}
    #     plotRes["class"] = ["bibliodendrogram"]
    #
    #     if (graph):
    #         plot(plotRes)
    #
    #     return plotRes
    X = np.random.rand(km_res, clust)
    return ff.create_dendrogram(X)


def plotCoord(g, side="b"):

    a = g.data

    ymin = np.min([np.min(l["y"]) for l in a if "y" in l], default=np.nan)
    ymax = np.max([np.max(l["y"]) for l in a if "y" in l], default=np.nan)
    xmin = np.min([np.min(l["x"]) for l in a if "x" in l], default=np.nan)
    xmax = np.max([np.max(l["x"]) for l in a if "x" in l], default=np.nan)

    coord = np.array([xmin, xmax, ymin, ymax])

    xl = np.array([xmax - 0.02 - (xmax - xmin) * 0.125, xmax - 0.02])

    if side == "b":
        yl = np.array([ymin, ymin + (ymax - ymin) * 0.125]) + 0.02
    else:
        yl = np.array([ymax - 0.02 - (ymax - ymin) * 0.125, ymax - 0.02])

    coord = np.concatenate((xl, yl))

    return coord


def eigCorrection(res):
    """
    Calculates Benzecri correction for eigenvalues and adds it to the given result object.

    Parameters:
        res: Result object containing eigenvalues to be corrected.

    Returns:
        Updated Result object with corrected eigenvalues added.
    """
    # Benzecri correction calculation
    n = res.eig.shape[0]

    e = res.eig[:, 0]
    eigBenz = ((n / (n - 1)) ** 2) * ((e - (1 / n)) ** 2)
    eigBenz[e < 1 / n] = 0
    perc = eigBenz / sum(eigBenz) * 100
    cumPerc = np.cumsum(perc)

    res.eigCorr = pd.DataFrame({'eig': e, 'eigBenz': eigBenz, 'perc': perc, 'cumPerc': cumPerc})

    return res



def euclDist(x, y):
    df = pd.DataFrame(np.nan, index=x.index, columns=y.index)
    for i in range(len(y)):
        ref = y.iloc[i, :2]
        df.iloc[:, i] = np.apply_along_axis(lambda r: np.sqrt(((r - ref) ** 2).sum()), axis=1, arr=x.iloc[:, :2])

    x['color'] = np.argmin(df.values, axis=1)
    return x




def factorial(X, method, quanti=[], quali=[]):
    df_quali = pd.DataFrame()
    df_quanti = pd.DataFrame()

    if method == "CA":
        res_mca = CA(n_components=2, copy=True, check_input=True, engine='auto',
                     random_state=None, quali_sup=quali, quanti_sup=quanti).fit(X)
        coord = res_mca.column_coordinates_
        df = pd.DataFrame(coord, columns=["Dim.1", "Dim.2"])
        if len(quali) > 0:
            df_quali = pd.DataFrame(res_mca.supplementary_row_coordinates_(quali))
        if len(quanti) > 0:
            df_quanti = pd.DataFrame(res_mca.supplementary_column_coordinates_(quanti))
        coord_doc = res_mca.row_coordinates_
        df_doc = pd.DataFrame(coord_doc)

    elif method == "MCA":
        if len(quanti) > 0:
            X.iloc[:, quanti] = X.iloc[:, quanti].apply(lambda x: pd.factorize(x)[0])
        else:
            X = X.apply(lambda x: pd.factorize(x)[0])
        res_mca = MCA(n_components=2, copy=True, check_input=True, engine='auto',
                      random_state=None, quali_sup=quali, quanti_sup=quanti).fit(X)
        coord = res_mca.column_coordinates_
        df = pd.DataFrame(coord[1::2, :], columns=["Dim.1", "Dim.2"])
        df.index = df.index.str.replace("_1", "")
        if len(quali) > 0:
            df_quali = pd.DataFrame(res_mca.supplementary_row_coordinates_(quali)[::2, :])
            df_quali.index = df_quali.index.str.replace("_1", "")
        if len(quanti) > 0:
            df_quanti = pd.DataFrame(res_mca.supplementary_column_coordinates_(quanti)[::2, :])
            df_quanti.index = df_quanti.index.str.replace("_1", "")
        coord_doc = res_mca.row_coordinates_
        df_doc = pd.DataFrame(coord_doc)

    elif method == "MDS":
        net_matrix = X.dot(X.T)
        net = 1 - normalize(net_matrix, axis=1, norm='l1')
        net[np.diag_indices_from(net)] = 0
        res_mca = MDS(n_components=2, dissimilarity='precomputed', random_state=None).fit(net)
        df = pd.DataFrame(res_mca.embedding_, columns=["Dim.1", "Dim.2"])
        df.index = X.index


def factorial(X, method, quanti, quali):
    df_quali = pd.DataFrame()
    df_quanti = pd.DataFrame()

    if method == 'CA':
        res_mca = prince.CA(n_components=2, n_iter=3, copy=True, check_input=True, engine='auto', random_state=42)
        res_mca = res_mca.fit(X)

        # Get coordinates of keywords
        coord = res_mca.column_coordinates(X)
        df = pd.DataFrame(coord.values, index=coord.index, columns=['dim1', 'dim2'])

        if quali is not None:
            df_quali = pd.DataFrame(res_mca.supplementary_row_columns(X)[1], index=coord.index,
                                    columns=['dim1', 'dim2'])

        if quanti is not None:
            df_quanti = pd.DataFrame(res_mca.supplementary_column_columns(X)[1], index=coord.columns,
                                     columns=['dim1', 'dim2'])

        coord_doc = res_mca.row_coordinates(X)
        docCoord = pd.DataFrame(coord_doc.values, index=coord_doc.index, columns=['dim1', 'dim2'])
        docCoord['contrib'] = coord_doc.cont_contrib.sum(axis=1)

    elif method == 'MCA':
        if quanti:
            X[X.columns[quanti]] = X[X.columns[quanti]].astype('category')
        else:
            X = X.astype('category')

        res_mca = prince.MCA(n_components=2, n_iter=3, copy=True, check_input=True, engine='auto', random_state=42)
        res_mca = res_mca.fit(X)

        # Get coordinates of keywords (we take only categories "1"")
        coord = res_mca.column_coordinates(X)
        df = pd.DataFrame(coord.values[1::2], index=coord.index[1::2], columns=['dim1', 'dim2'])
        df.index = df.index.str.replace('_1', '')

        if quali is not None:
            df_quali = pd.DataFrame(res_mca.supplementary_row_columns(X)[1][1::2], index=coord.index[1::2],
                                    columns=['dim1', 'dim2'])
            df_quali.index = df_quali.index.str.replace('_1', '')

        if quanti is not None:
            df_quanti = pd.DataFrame(res_mca.supplementary_column_columns(X)[1][1::2], index=coord.columns[1::2],
                                     columns=['dim1', 'dim2'])
            df_quanti.index = df_quanti.index.str.replace('_1', '')

        coord_doc = res_mca.row_coordinates(X)
        docCoord = pd.DataFrame(coord_doc.values, index=coord_doc.index, columns=['dim1', 'dim2'])
        docCoord['contrib'] = coord_doc.cont_contrib.sum(axis=1)

    elif method == 'MDS':
        NetMatrix = np.dot(X.T, X)
        Net = 1 - normalize(NetMatrix, norm='l2', axis=1, copy=True)
        np.fill_diagonal(Net, 0)
        res_mca = manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        res_mca = res_mca.fit(Net)

        df = pd.DataFrame(res_mca.embedding_, index=Net.index, columns=['dim1', 'dim2'])

    else:
        raise ValueError('Invalid method argument')

    results = {'res.mca': res_mca, 'df': df, 'df_doc': docCoord, 'df_quali': df_quali, 'df_quanti': df_quanti,'docCoord' : docCoord }



    return (results)