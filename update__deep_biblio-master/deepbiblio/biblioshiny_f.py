# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
import os
import webbrowser
#from bibliometrix import app

def biblioshiny(host="127.0.0.1", port=None, launch_browser=True, max_upload_size=200):
    os.environ["shiny.host"] = host
    #app.shinyOptions.maxUploadSize = max_upload_size
    if launch_browser:
        webbrowser.open("http://{0}:{1}".format(host, port))
    #app.run(host=host, port=port, debug=True, threaded=True)