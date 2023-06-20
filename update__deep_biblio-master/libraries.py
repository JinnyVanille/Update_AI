# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

# import shinyswatch

# import qgrid
# from shinywidgets import *

# import textwrap
# from pyBibX.base import pbx_probe
# from google.colab import data_table

# Widgets
# from ipydatagrid import DataGrid


# def messageItem2(from_, message, icon=shiny.icon("user"), time=None, href=None, inputId=None):
#     if href is None:
#         href = "#"
#     return shiny.tags.li(shiny.tags.a(id=inputId, class_="action-button" if inputId else None,
#                                       href=href, target="_blank", children=[icon,
#                                       shiny.tags.h4(from_, shiny.tags.small(shiny.icon("clock-o"), time)) if time else None,
#                                       shiny.tags.p(message)]))

def initial(values):
    values["results"] = ["NA"]
    values["log"] = "working..."
    values["load"] = "FALSE"
    values["field"] = values["cocngrams"] = "NA"
    values["citField"] = values["colField"] = values["citSep"] = "NA"
    values["NetWords"] = values["NetRefs"] = values["ColNetRefs"] = [[float('nan')]]
    values["Title"] = "Network"
    values["Histfield"] = "NA"
    values["histlog"] = "working..."
    values["kk"] = 0
    values["histsearch"] = "NA"
    values["citShortlabel"] = "NA"
    values["S"] = ["NA"]
    values["GR"] = "NA"

    return values
