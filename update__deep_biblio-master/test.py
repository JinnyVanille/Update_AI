# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

import shinyswatch
from shiny import ui, App
from shinywidgets import output_widget, render_widget
import plotly.express as px
import plotly.graph_objs as go

import shiny
from shiny import *
# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#

from shiny.types import *

df = px.data.tips()

app_ui = ui.page_fluid(
    shinyswatch.theme.superhero(),
    ui.div(
        ui.input_select(
            "x", label="Variable",
            choices=["total_bill", "tip", "size"]
        ),
        ui.input_select(
            "color", label="Color",
            choices=["smoker", "sex", "day", "time"]
        ),
        class_="d-flex gap-3"
    ),
    output_widget("my_widget")
)

def server(input, output, session):
    @output
    @render_widget
    def my_widget():
        fig = px.histogram(
            df, x=input.x(), color=input.color(),
            marginal="rug"
        )
        fig.layout.height = 275
        return fig

app = App(app_ui, server)

if __name__ == '__main__':
    run_app(app='test:app', host='127.0.0.1', port=8000, autoreload_port=0, reload=True, ws_max_size=16777216,
            log_level=None, app_dir='.', factory=False, launch_browser=True)