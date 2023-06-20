# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
from deepbiblio.convert2df_f import convert2df
from deepbiblio.metaTagExtraction_f import *
from deepbiblio.tableTag_f import *
from libraries import *
from utils import *
from userinterface import *
import copy
from pathlib import Path

values = {}

def print_start(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{name}')  # Press Ctrl+F8 to toggle the breakpoint.
    run_app(app='app:app', host='127.0.0.1', port=8000, autoreload_port=0, reload=True, ws_max_size=16777216, log_level=None, app_dir='.', factory=False, launch_browser=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_start('Deep Biblio is starting...')

def server(input: Inputs, output: Outputs, session: Session):
    global values
#    session.onSessionEnded(stop_app())
    ## suppress warnings
    # warnings.filterwarnings("ignore")

    ## file upload max size
    maxUploadSize = 2000  # default value
    # maxUploadSize = getShinyOption("maxUploadSize", maxUploadSize)
    # options(shiny.maxRequestSize = maxUploadSize * 1024 ** 2)

    ## initial values
    # # values["sidebar"] = sidebarMenu()
    values["rest_sidebar"] = False
    values["list_file"] = pd.DataFrame({"sheet": [1], "file": [2], "n": [3]})
    # values["wb"] = openpyxl.workbook()
    # # values["dfLabel"] = dfLabel()
    values["myChoices"] = "Empty Report"
    # # values["logo"] = logo
    # # values["logoGrid"] = grid.rasterGrob(logo, interpolate=True)
    #
    # ### setting values
    values["dpi"] = 300
    values["h"] = 7
    values["w"] = 14
    values["path"] = os.getcwd() + "/"
    # ###
    #
    values["results"] = ["NA"]
    values["log"] = "working..."
    values["load"] = "FALSE"
    values["field"] = values["cocngrams"] = "NA"
    values["citField"] = values["colField"] = values["citSep"] = "NA"
    values["NetWords"] = values["NetRefs"] = values["ColNetRefs"] = [[None]]
    values["Title"] = "Network"
    values["Histfield"] = "NA"
    values["histlog"] = "working..."
    values["kk"] = 0
    values["M"] = pd.DataFrame({"PY": [1]})
    values["Source title"] = pd.DataFrame({"PY": [1]})
    values["histsearch"] = "NA"
    values["citShortlabel"] = "NA"
    values["S"] = ["NA"]
    values["GR"] = "NA"
    values["dsToken"] = "Wrong account or password"
    values["dsSample"] = 0
    values["dsQuery"] = ""
    values["pmQuery"] = " "
    values["pmSample"] = 0
    values["ApiOk"] = 0
    values["checkControlBar"] = False

    @output
    @render.ui
    def contents():
        if input.file1() is None:
            return "Please upload a file"
        f: list[FileInfo] = input.file1()
        df = pd.read_csv(f[0]["datapath"], header=0 if input.header() else None)
        return ui.HTML(df.to_html(classes="table table-striped"))

    @output
    @render_widget
    def my_widget():
        df = px.data.tips()
        fig = px.histogram(
            df, x=input.x(), color=input.color(),
            marginal="rug"
        )
        fig.layout.height = 275
        return fig

    @output(id="figure")
    @render.ui
    def _():
        return output_widget(input.framework())

    @output(id="ipyleaflet")
    @render_widget
    def _():
        from ipyleaflet import Map, Marker

        m = Map(center=(52.204793, 360.121558), zoom=4)
        m.add_layer(Marker(location=(52.204793, 360.121558)))
        return m

    @output(id="qgrid")
    @render_widget
    def _():
        import qgrid

        randn = np.random.randn
        df_types = pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        "2013-01-01",
                        "2013-01-02",
                        "2013-01-03",
                        "2013-01-04",
                        "2013-01-05",
                        "2013-01-06",
                        "2013-01-07",
                        "2013-01-08",
                        "2013-01-09",
                    ],
                    index=list(range(9)),
                    dtype="datetime64[ns]",
                ),
                "B": pd.Series(randn(9), index=list(range(9)), dtype="float32"),
                "C": pd.Categorical(
                    [
                        "washington",
                        "adams",
                        "washington",
                        "madison",
                        "lincoln",
                        "jefferson",
                        "hamilton",
                        "roosevelt",
                        "kennedy",
                    ]
                ),
                "D": [
                    "foo",
                    "bar",
                    "buzz",
                    "bippity",
                    "boppity",
                    "foo",
                    "foo",
                    "bar",
                    "zoo",
                ],
            }
        )
        df_types["E"] = df_types["D"] == "foo"
        return qgrid.show_grid(df_types, show_toolbar=True)

    # @output(id="altair")
    # @render_widget
    # def _():
    #     import altair as alt
    #     from vega_datasets import data
    #
    #     return (
    #         alt.Chart(data.cars())
    #             .mark_point()
    #             .encode(
    #             x="Horsepower",
    #             y="Miles_per_Gallon",
    #             color="Origin",
    #         )
    #     )

    @output(id="plotly")
    @render_widget
    def _():
        import plotly.graph_objects as go

        return go.FigureWidget(
            data=[go.Bar(y=[2, 1, 3])],
            layout_title_text="A Figure Displayed with fig.show()",
        )

    @output(id="bqplot")
    @render_widget
    def _():
        from bqplot import Axis, Bars, Figure, LinearScale, Lines, OrdinalScale

        size = 20
        x_data = np.arange(size)
        scales = {"x": OrdinalScale(), "y": LinearScale()}

        return Figure(
            title="API Example",
            legend_location="bottom-right",
            marks=[
                Bars(
                    x=x_data,
                    y=np.random.randn(2, size),
                    scales=scales,
                    type="stacked",
                ),
                Lines(
                    x=x_data,
                    y=np.random.randn(size),
                    scales=scales,
                    stroke_width=3,
                    colors=["red"],
                    display_legend=True,
                    labels=["Line chart"],
                ),
            ],
            axes=[
                Axis(scale=scales["x"], grid_lines="solid", label="X"),
                Axis(
                    scale=scales["y"],
                    orientation="vertical",
                    tick_format="0.2f",
                    grid_lines="solid",
                    label="Y",
                ),
            ],
        )

    @output(id="ipychart")
    @render_widget
    def _():
        from ipychart import Chart

        dataset = {
            "labels": [
                "Data 1",
                "Data 2",
                "Data 3",
                "Data 4",
                "Data 5",
                "Data 6",
                "Data 7",
                "Data 8",
            ],
            "datasets": [{"data": [14, 22, 36, 48, 60, 90, 28, 12]}],
        }

        return Chart(data=dataset, kind="bar")

    @output(id="ipywebrtc")
    @render_widget
    def _(ipywebrtc=None):
        from ipywebrtc import CameraStream

        return CameraStream(
            constraints={
                "facing_mode": "user",
                "audio": False,
                "video": {"width": 640, "height": 480},
            }
        )

    @output(id="ipyvolume")
    @render_widget
    def _():
        from ipyvolume import quickquiver

        x, y, z, u, v, w = np.random.random((6, 1000)) * 2 - 1
        return quickquiver(x, y, z, u, v, w, size=5)

    @output(id="pydeck")
    @render_widget
    def _():
        import pydeck as pdk

        UK_ACCIDENTS_DATA = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv"

        layer = pdk.Layer(
            "HexagonLayer",  # `type` positional argument is here
            UK_ACCIDENTS_DATA,
            get_position=["lng", "lat"],
            auto_highlight=True,
            elevation_scale=50,
            pickable=True,
            elevation_range=[0, 3000],
            extruded=True,
            coverage=1,
        )

        # Set the viewport location
        view_state = pdk.ViewState(
            longitude=-1.415,
            latitude=52.2323,
            zoom=6,
            min_zoom=5,
            max_zoom=15,
            pitch=40.5,
            bearing=-27.36,
        )

        # Combined all of it and render a viewport
        return pdk.Deck(layers=[layer], initial_view_state=view_state)

    @output(id="bokeh")
    @render_widget
    def _():
        from bokeh.plotting import figure

        x = [1, 2, 3, 4, 5]
        y = [6, 7, 2, 4, 5]
        p = figure(title="Simple line example", x_axis_label="x", y_axis_label="y")
        p.line(x, y, legend_label="Temp.", line_width=2)
        return p

    def format_function(obj):
        ext = obj[0].split('.')[-1]
        if ext == 'txt':
            format = "plaintext"
        elif ext == 'csv':
            format = "csv"
        elif ext == 'bib':
            format = "bibtex"
        elif ext == 'ciw':
            format = "endnote"
        elif ext == 'xlsx':
            format = "excel"
        return format

    @output
    @render.table
    @reactive.Calc
    @reactive.event(input.load_data_start_action)
    def applyLoad():
        global values
        # input$file1 will be None initially. After the user selects
        # and uploads a file, it will be a data frame with 'name',
        # 'size', 'type', and 'datapath' columns. The 'datapath'
        # column will contain the local filenames where the data can
        # be found.
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")

            p.set(12, message="Computing")
            if input['load'].get() == "demo":
                management = pd.read_csv("data/Bibliometrix-Export-File-Management.csv")
                p.set(14, message="Computing")
                values = initial(values)
                management.index = management['SR']
                values['M'] = management
                values['Morig'] = management
                values['Histfield'] = "NA"
                values['results'] = list("NA")
                values['rest_sidebar'] = True
                d = copy.deepcopy(values['M'])
                p.set(15, message="Computing")
                return pd.DataFrame(list(d.items()), columns=['Attributes', 'Values'])
            inFile = input['file1']
            p.set(13, message="Computing")
            if inFile is not None and input['load'].get() == "import":
                ext = "." + get_file_extension(inFile.get()[0]['datapath'])
                if input['dbsource'].get() == "isi":
                    if ext == ".zip":
                        D = ZipFile(inFile['datapath']).extractall()
                        M = convert2df(file=D, dbsource=input['dbsource'].get(), format=format_function(get_file_extension(D)))
                    else:
                        M = convert2df(file=inFile.get()[0]['datapath'], dbsource=input['dbsource'].get(), format=format_function(ext))
                elif input['dbsource'].get() == 'scopus':
                    if ext == ".zip":
                        D = ZipFile(inFile['datapath']).extractall()
                        M = convert2df(file=D, dbsource=input['dbsource'].get(), format=format_function(D))
                    elif ext == ".csv":
                        M = convert2df(file=inFile.get()[0]['datapath'], dbsource=input['dbsource'].get(), format="csv")
                    elif ext == ".bib":
                        M = convert2df(file=inFile.get()[0]['datapath'], dbsource=input['dbsource'].get(), format="bibtex")

            elif inFile is not None and input['load'].get() == "load":
                ext = get_file_extension(inFile.get()[0]['datapath'])
                if ext == '.xlsx':
                    M = pd.read_excel(inFile['datapath'], col_types="text")
                    M['PY'] = pd.to_numeric(M['PY'])
                    M['TC'] = pd.to_numeric(M['TC'])
                    M = M.astype({"SR": str})
                    SR = M['SR']
                    tab = SR.value_counts()
                    tab2 = tab.value_counts()
                    ind = [int(name) for name in tab2.index if int(name) > 1]
                    if len(ind) > 0:
                        for i in ind:
                            indice = tab[tab == i].index.tolist()
                            for j in indice:
                                indice2 = SR[SR == j].index.tolist()
                                SR[indice2] = [f"{x} {k}" for k, x in enumerate(SR[indice2], start=1)]
                    M['SR'] = SR
                    M = M.set_index('SR')
                elif ext in ('.rdata', '.rda'):
                    M = pd.read_pickle(inFile.get()[0]['datapath'])
                elif ext == '.rds':
                    M = pd.read_rds(inFile.get()[0]['datapath'])
            elif inFile is None:
                return None
            p.set(14, message="Computing")
            # remove not useful columns
            ind = [i for i, name in enumerate(M.columns) if name.startswith('X.')]
            if len(ind) > 0:
                M = M.drop(M.columns[ind], axis=1)

            values = initial(values)
            values["M"] = M
            values['Morig'] = M
            values['Histfield'] = "NA"
            values['results'] = ["NA"]

            if len(M.columns) > 1:
                values['rest_sidebar'] = True
            #if len(M.columns) > 1:
            #    showModal
            p.set(15, message="Computing")
            d = copy.deepcopy(M)
            return pd.DataFrame(list(d.items()), columns=['Attributes', 'Values'])

    @output(id="sources_most_relevant_sources_baseline_plot")
    @render.ui
    @reactive.Calc
    @reactive.event(input.sources_most_relevant_sources_start_action)
    def _():
        return output_widget("sources_most_relevant_sources_select_data_plot")

    @output(id="sources_most_relevant_sources_select_data_plot")
    @render_widget
    def _():
        global values
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")
            p.set(5, message="Computing")
            res = descriptive(values, type="tab7")
            p.set(14, message="Computing")
            values = res["values"]
            values["TABSo"] = values["TAB"]
            xx = values["TAB"].dropna()
            if input.sources_most_relevant_sources_input_numeric() > xx.shape[0]:
                  k = xx.shape[0]
            else:
                  k = input.sources_most_relevant_sources_input_numeric()
            xx = xx.sort_values("Number of document", ascending=False).head(k)
            xx["Source title"] = xx["Source title"].str[:50]
            g = px.bar(xx, x="Number of document", y="Source title", orientation="h",
                       title="Most Relevant Sources", text="Number of document")
            g.update_layout(yaxis=dict(autorange="reversed"))
            values["MRSplot"] = g
            p.set(15, message="Computing")
        return g

    @output(id="sources_most_relevant_sources_baseline_table")
    @render.ui
    @reactive.Calc
    @reactive.event(input.sources_most_relevant_sources_start_action)
    def _():
        return output_widget("sources_most_relevant_sources_select_data_table")

    @output(id="sources_most_relevant_sources_select_data_table")
    @render_widget
    def _():
        global values
        res = descriptive(values, type="tab7")
        values = res["values"]
        values["TABSo"] = values["TAB"]
        xx = values["TAB"].dropna()
        if input.sources_most_relevant_sources_input_numeric() > xx.shape[0]:
            k = xx.shape[0]
        else:
            k = input.sources_most_relevant_sources_input_numeric()
        xx = xx.sort_values("Number of document", ascending=False).head(k)
        xx["Source title"] = xx["Source title"].str[:50]

        xx = xx.loc[:, xx.columns.intersection(['Source title', 'Number of document'])]
        import qgrid
        g = qgrid.show_grid(xx, show_toolbar=False)
        values["MRSplot"] = g
        return g

############################
    

    @output(id="sources_most_local_cited_sources_baseline_plot")
    @render.ui
    @reactive.Calc
    @reactive.event(input.sources_most_local_cited_sources_start_action)
    def _():
        return output_widget("sources_most_local_cited_sources_baseline_data_plot")

    @output(id="sources_most_local_cited_sources_baseline_data_plot")
    @render_widget
    def _():
        global values
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")
            p.set(5, message="Computing")
            M = metaTagExtraction(values["M"], "CR_SO")
            p.set(13, message="Computing")

            values['M'] = metaTagExtraction(values['M'], 'CR_SO')
            TAB = tableTag(values['M'], 'CR_SO')
            TAB = pd.DataFrame({'Sources': list(TAB.keys()), 'Articles': list(TAB.values)})
            values['TABSoCit'] = TAB
            xx = TAB
            if input.sources_most_local_cited_sources_input_numeric() > xx.shape[0]:
                k = xx.shape[0]
            else:
                k = input.sources_most_local_cited_sources_input_numeric()
            xx = xx.loc[xx.index[0:k], :]
            xx['Articles'] = pd.to_numeric(xx['Articles'])
            xx["Sources"] = xx["Sources"].str[:50]
            xx = xx.sort_values("Articles", ascending=False).head(k)
            # Create and return frequency plot
            fig = px.bar(xx, x='Articles', y='Sources', orientation='h',
                         title='Most Local Cited Sources',
                         text=xx.columns[1],
                         labels={'Sources': 'Cited Sources', 'Articles': 'N. of Local Citations'})
            fig.update_layout(yaxis=dict(autorange="reversed"))
            p.set(15, message="Computing")
            return fig

    @output(id="sources_most_local_cited_sources_baseline_table")
    @render.ui
    @reactive.Calc
    @reactive.event(input.sources_most_relevant_sources_start_action)
    def _():
        return output_widget("sources_most_local_cited_sources_select_data_table")

    @output(id="sources_most_local_cited_sources_select_data_table")
    @render_widget
    def _():
        global values
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")
            p.set(5, message="Computing")
            M = metaTagExtraction(values["M"], "CR_SO")
            p.set(13, message="Computing")

            values['M'] = metaTagExtraction(values['M'], 'CR_SO')
            TAB = tableTag(values['M'], 'CR_SO')
            TAB = pd.DataFrame({'Sources': list(TAB.keys()), 'Articles': list(TAB.values)})
            values['TABSoCit'] = TAB
            xx = TAB
            if input.sources_most_local_cited_sources_input_numeric() > xx.shape[0]:
                k = xx.shape[0]
            else:
                k = input.sources_most_local_cited_sources_input_numeric()
            xx = xx.loc[xx.index[0:k], :]
            xx['Articles'] = pd.to_numeric(xx['Articles'])
            xx["Sources"] = xx["Sources"].str[:50]
            xx = xx.sort_values("Articles", ascending=False).head(k)
            # Create and return data table
            xx = xx.loc[:, xx.columns.intersection(['Sources', 'Articles'])]
            import qgrid
            g = qgrid.show_grid(xx, show_toolbar=False)
            values["MRSplot"] = g
            return g

############################

    @output(id="authors_most_relevant_authors_baseline_plot")
    @render.ui
    @reactive.Calc
    @reactive.event(input.authors_most_relevant_authors_start_action)
    def _():
        return output_widget("authors_most_relevant_authors_baseline_plot_data_plot")

    @output(id="authors_most_relevant_authors_baseline_plot_data_plot")
    @render_widget
    def _():
        global values
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")
            p.set(5, message="Computing")
            res = descriptive(values, type="tab3")
            p.set(13, message="Computing")
            values = res["values"]
            values["ABAu"] = values["TAB"]

            xx = values["ABAu"]
            if input.author_frequency_measure() == 't':
                lab = 'N. of Documents'
                xx = xx.iloc[:, 0:2]
            elif input.author_frequency_measure() == 'p':
                xx = xx.iloc[:, 0:2]
                xx.iloc[:, 1] = pd.to_numeric(xx.iloc[:, 1]) / values['M'].shape[0] * 100
                lab = 'N. of Documents (in %)'
            else:
                xx = xx.iloc[:, [0, 2]]
                lab = 'N. of Documents (Fractionalized)'

            xx.iloc[:, 1] = pd.to_numeric(xx.iloc[:, 1])

            if input.authors_most_relevant_authors_input_numeric() > xx.shape[0]:
                k = xx.shape[0]
            else:
                k = input.authors_most_relevant_authors_input_numeric()

            xx = xx.sort_values(by=[xx.columns[1]], ascending=False).iloc[0:k, :]
            xx.iloc[:, 1] = round(xx.iloc[:, 1], 1)
            p.set(14, message="Computing")
            fig = px.bar(xx, x=xx.columns[1], y=xx.columns[0], orientation='h', text=xx.columns[1],
                         title='Most Relevant Authors', labels={xx.columns[1]: lab, xx.columns[0]: 'Authors'})
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(yaxis=dict(autorange="reversed"))
            p.set(15, message="Computing")
        return fig

    @output(id="authors_most_relevant_authors_baseline_table")
    @render.ui
    @reactive.Calc
    @reactive.event(input.authors_most_relevant_authors_start_action)
    def _():
        return output_widget("authors_most_relevant_authors_select_data_table")

    @output(id="authors_most_relevant_authors_select_data_table")
    @render_widget
    def _():
        global values
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")
            p.set(5, message="Computing")
            res = descriptive(values, type="tab3")
            p.set(13, message="Computing")
            values = res["values"]
            values["ABAu"] = values["TAB"]
            xx = values["ABAu"]
            if input.author_frequency_measure() == 't':
                lab = 'N. of Documents'
                xx = xx.iloc[:, 0:2]
            elif input.author_frequency_measure() == 'p':
                xx = xx.iloc[:, 0:2]
                xx.iloc[:, 1] = pd.to_numeric(xx.iloc[:, 1]) / values['M'].shape[0] * 100
                lab = 'N. of Documents (in %)'
            else:
                xx = xx.iloc[:, [0, 2]]
                lab = 'N. of Documents (Fractionalized)'

            xx.iloc[:, 1] = pd.to_numeric(xx.iloc[:, 1])

            if input.authors_most_relevant_authors_input_numeric() > xx.shape[0]:
                k = xx.shape[0]
            else:
                k = input.authors_most_relevant_authors_input_numeric()

            xx = xx.sort_values(by=[xx.columns[1]], ascending=False).iloc[0:k, :]
            xx.iloc[:, 1] = round(xx.iloc[:, 1], 1)
            indexList = list()
            for x in range(len(xx["Authors"])):
                indexList.append(x)
            xx["index"] = indexList
            p.set(14, message="Computing")
            # Create and return data table
            xx.columns = ["Authors", lab, "index"]
            # xx = xx.loc[:, xx.columns.intersection(["index", "Authors", lab])]
            import qgrid
            g = qgrid.show_grid(xx, show_toolbar=False)
            values["MRSplot"] = g
            return g

############################

www_dir = Path(__file__).parent / "www"
app = App(app_ui(), server, static_assets=www_dir)

