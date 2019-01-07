import tkinter as tk
from constants import *


# statusbar for messages to user
class TkStatusBar(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master.root)
        self.label = tk.Label(self, bd=0, bg=GUI_BG, relief="sunken", anchor="w")
        self.label.pack(fill="x")
        self.init()

    # initialize the status bar with default value.
    def init(self):
        # message = "Copyright 2018, Nathan de Bruijn"
        message = ""
        self.label.config(text="    " + message)

    # change the value of the statusbar to the value of text
    def set(self, text, *args):
        self.label.config(text="    " + text % args)
        self.label.update_idletasks()

    # clear the statusbar
    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()
