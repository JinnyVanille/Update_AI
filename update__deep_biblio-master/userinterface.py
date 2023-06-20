# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

import shinyswatch
from shiny import ui, App, html_dependencies
from shinywidgets import *
import plotly.express as px
import plotly.graph_objs as go

import shiny
from shiny import *
from shiny.types import *


def description_analysis_layout():
    return ui.tags.iframe(src="http://127.0.0.1:5000/", width="100%", height="1000px")


def app_ui():
    return ui.page_navbar(
        bokeh_dependency(),
        # Available themes:
        #  cerulean, cosmo, cyborg, darkly, flatly, journal, litera, lumen, lux,
        #  materia, minty, morph, pulse, quartz, sandstone, simplex, sketchy, slate,
        #  solar, spacelab, superhero, united, vapor, yeti, zephyr
        # The shinyswatch package provides themes from https://bootswatch.com/
        shinyswatch.theme.superhero(),
        ui.nav(
            "Main",
            ui.panel_main(
                ui.tags.h3("Deep Biblio: Deep Learning Bibliometric Analysis", style="text-align: center;"),
                ui.tags.img(src="deepbiblio.jpg",
                            style="display: block;margin-left: auto;margin-right: auto;width: 50%;"),
                ui.tags.h6(
                    "Deep Biblio is an open-source and freely available for use, distributed under the MIT license.",
                    style="text-align: center;"),
                ui.tags.h6(
                    "When Deep Biblio software are used in a publication, authors are required to cite the following reference:",
                    style="text-align: center;"),
                ui.tags.h6(
                    ui.tags.em(ui.tags.b(
                        "Tiong Yew Tang (2023). Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis. Journal of Informetics.",
                        )), style="text-align: center;"),
                ui.tags.h6(
                    "To honor the time and efforts of our contributing members. Hence, failure to cite the software is a violation of the license.",
                    style="text-align: center;"),
                ui.tags.h6("Project Leader: Tang Tiong Yew", style="text-align: center;"),
                ui.tags.div(
                    ui.tags.a("email: tiongyewt@sunway.edu.my", href="mailto:tiongyewt@sunway.edu.my;", align="center"),
                    style="text-align: center;"),
                ui.tags.h6("Tang Tiong Yew, Copyright 2023", style="text-align: center;"),
                width=12,
            ),

        ),
        # ui.nav_menu(
        #     "Data",
        ui.nav(
            "Load Data",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.tags.h5("Import Data:"),
                    ui.input_select("load", "Choose your action:", {"demo": "Sample File", "import": "Raw File",
                                                                    "load": "Deep Biblio Excel File (xlsx)"},
                                    selected=None),
                    ui.panel_conditional(
                        "input.load === 'import'",
                        ui.input_select("dbsource", "Database:", {"scopus": "Scopus", "ISI": "Web of Science",
                                                                  "dimensions": "Dimensions",
                                                                  "lens": "Lens.org", "pubmed": "Pub Med",
                                                                  "cochranelibrary": "Cochrane Library"},
                                        selected=None),
                    ),
                    ui.panel_conditional(
                        "input.load !== '' && input.load !== 'demo'",
                        ui.input_file("file1", "File input:", multiple=False,
                                      accept=[".csv", ".txt", ".ciw", ".bib",
                                              ".xlsx", ".zip", ".xls", ".pkl", ".pickle"]
                                      )
                    ),
                    ui.panel_conditional(
                        "input.load !== ''",
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "load_data_start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.tags.h5("Export Data:"),
                    ui.input_select("save_file", "Save file as:",
                                    {"": "", "xlsx": "Excel", "pickle": "Python Pickle Data"}, selected=None),
                    ui.panel_conditional(
                        "input.save_file !== ''",
                        ui.input_action_button(
                            "export_data_save_action", "Save", class_="btn-primary"
                        ),
                    ),
                    width=3,
                ),
                ui.panel_main(
                    ui.navset_bar(
                        ui.nav(
                            ui.output_table("applyLoad"),
                        ),
                        title="",
                    ),
                    width=9
                ),
            ),
            # ),
            # ui.nav(
            #     "Access Data",
            #     ui.layout_sidebar(
            #         ui.panel_sidebar(
            #             ui.tags.h5("Import Data:"),
            #         ),
            #         ui.panel_main(
            #             ui.navset_tab(
            #                 ui.nav(
            #                     "Tab 1",
            #                     ui.tags.h4("Table"),
            #                 ),
            #                 ui.nav("Tab 2")
            #             )
            #         ),
            #     ),
            # ),
        ),
        ui.nav(
            "Filters",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.tags.h5("Import Data:")
                ),
                ui.panel_main(
                    ui.navset_tab(
                        ui.nav(
                            "Plot",
                            ui.tags.h4("Table"),
                        ),
                        ui.nav("Table")
                    )
                ),
            ),
        ),
        ui.nav_menu(
            "Descriptive",
            ui.nav(
                "Primary Information",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Yearly Scientific Production",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Yearly Average Citation",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Three-Field Plot",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Sources",
            ui.nav(
                "Most Relevant Sources",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click start to generate display:"),
                        ui.input_numeric("sources_most_relevant_sources_input_numeric", label="Number of sources:",
                                         value=10),
                        ui.input_action_button(
                            "sources_most_relevant_sources_start_action", "Start", class_="btn-primary"
                        ),
                        width=3,
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Baseline Plot",
                                ui.output_ui("sources_most_relevant_sources_baseline_plot"),
                            ),
                            ui.nav(
                                "Baseline Table",
                                ui.output_ui("sources_most_relevant_sources_baseline_table"),
                            ),
                            ui.nav(
                                "Deep Learning Plot",
                                ui.output_ui("sources_most_relevant_sources_deep_learning_plot"),
                            ),
                            ui.nav(
                                "Deep Learning Table",
                                ui.output_ui("sources_most_relevant_sources_deep_learning_table"),
                            ),
                            ui.nav(
                                "Description Analysis",
                                description_analysis_layout(),
                            ),
                        ),
                        width=9
                    ),
                ),
            ),
            ui.nav(
                "Most Local Cited Sources",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click start to generate display:"),
                        ui.input_numeric("sources_most_local_cited_sources_input_numeric", label="Number of sources:", value=10),
                        ui.input_action_button(
                            "sources_most_local_cited_sources_start_action", "Start", class_="btn-primary"
                        ),
                        width=3,
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Baseline Plot",
                                ui.output_ui("sources_most_local_cited_sources_baseline_plot"),
                            ),
                            ui.nav(
                                "Baseline Table",
                                ui.output_ui("sources_most_local_cited_sources_baseline_table"),
                            ),
                            ui.nav(
                                "Deep Learning Plot",
                                ui.output_ui("sources_most_local_cited_sources_deep_learning_plot"),
                            ),
                            ui.nav(
                                "Deep Learning Table",
                                ui.output_ui("sources_most_local_cited_sources_deep_learning_table"),
                            ),
                            ui.nav(
                                "Description Analysis",
                                ui.output_ui("sources_most_local_cited_sources_description_analysis"),
                            ),
                        ),
                        width=9
                    ),
                ),
            ),
            ui.nav(
                "Bradford's Law",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click start to generate display:"),
                        ui.input_numeric("bradford_law_input_numeric", label="Number of source (n):",
                                         value=10),
                        ui.input_action_button(
                            "sources_bradford_law_start_action", "Start", class_="btn-primary"
                        ),
                        width=3,
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Baseline Plot",
                                ui.output_ui("sources_bradford_law_baseline_plot"),
                            ),
                            ui.nav(
                                "Deep Learning Plot",
                                ui.output_ui("sources_bradford_law_deep_learning_plot"),
                            ),
                            ui.nav(
                                "Baseline Table",
                                ui.output_ui("sources_bradford_law_baseline_table"),
                            ),
                            ui.nav(
                                "Deep Learning Table",
                                ui.output_ui("sources_bradford_law_deep_learning_table"),
                            ),
                        ),
                        width=9
                    ),
                ),
            ),
            ui.nav(
                "Sources' Production Over Time",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click start to load data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Sources' Local Impact",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click start to load data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Python Display Package Demo",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.div(
                            ui.input_radio_buttons(
                                "framework",
                                "Choose an ipywidget package",
                                [
                                    "qgrid",
                                    "ipyleaflet",
                                    "pydeck",
                                    # "altair",
                                    "plotly",
                                    "bokeh",
                                    "bqplot",
                                    "ipychart",
                                    "ipywebrtc",
                                    # TODO: fix me
                                    # "ipyvolume",
                                ],
                                selected="ipyleaflet",
                            )
                        ),
                        width=3
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.output_ui("figure"),
                            ),
                            ui.nav("Table")
                        ),
                        width=9
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Authors",
            ui.nav(
                "Most Relevant Authors",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click start to generate display:"),
                        ui.input_numeric("authors_most_relevant_authors_input_numeric", label="Number of authors:",
                                         value=10),
                        ui.input_select("author_frequency_measure", "Frequency measure:",
                                        {"t": "N. of Documents", "p": "Percentange",
                                         "f": "Fractionalized Frequency"},
                                        selected="t"),
                        ui.input_action_button(
                            "authors_most_relevant_authors_start_action", "Start", class_="btn-primary"
                        ),
                        width=3,
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Baseline Plot",
                                ui.output_ui("authors_most_relevant_authors_baseline_plot"),
                            ),
                            ui.nav(
                                "Baseline Table",
                                ui.output_ui("authors_most_relevant_authors_baseline_table"),
                            ),
                            ui.nav(
                                "Deep Learning Plot",
                                ui.output_ui("authors_most_relevant_authors_deep_learning_plot"),
                            ),
                            ui.nav(
                                "Deep Learning Table",
                                ui.output_ui("authors_most_relevant_authors_deep_learning_table"),
                            ),
                            ui.nav(
                                "Description Analysis",
                                ui.output_ui("authors_most_relevant_authors_description_analysis"),
                            ),
                        ),
                        width=9
                    ),
                ),
            ),
            ui.nav(
                "Most Local Cited Authors",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click start to load data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Authors' Production over Time",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Lotka's Law",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Authors' Local Impact",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Affiliations",
            ui.nav(
                "Most Relevant Affiliations",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Affiliations' Production Over Time",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Countries",
            ui.nav(
                "Corresponding Author's Countries",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Countries' Scientific Production",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Countries Production Over Time",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Most Cited Countries",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Documents",
            ui.nav(
                "Most Global Cited Documents",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Most Local Cited Documents",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Most Local Cited References",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "References Spectroscopy",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Most Frequent Words",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "WordCloud",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "TreeMap",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Words' Frequency Over Time",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Trend Topics",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Click Start to Load Data:"),
                        ui.input_action_button(
                            "start_action", "Start", class_="btn-primary"
                        ),
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Clustering",
            ui.nav(
                "Clustering by Coupling",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Conceptual Structure",
            ui.nav(
                "Co-occurence Network",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Thematic Map",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Thematic Evolution",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Factorial Analysis",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Intellectual Structure",
            ui.nav(
                "Co-citation Network",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Historiograph",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav_menu(
            "Social Structure",
            ui.nav(
                "Collaboration Network",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
            ui.nav(
                "Countries' Collaboration World Map",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.tags.h5("Import Data:")
                    ),
                    ui.panel_main(
                        ui.navset_tab(
                            ui.nav(
                                "Plot",
                                ui.tags.h4("Table"),
                            ),
                            ui.nav("Table")
                        )
                    ),
                ),
            ),
        ),
        ui.nav("Setting"),
        title=ui.tags.a("Deep Biblio", href="http://127.0.0.1:8000/",
                        style="nav-link active;text-decoration: none;color: inherit;")
    )
