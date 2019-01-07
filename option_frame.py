import tkinter as tk
from constants import GUI_BG, GUI_GRAYD, GUI_SHADOW, GUI_BLUE, STORE_N, button_config

LABEL_TITLE = {"bg": GUI_BG, "fg": GUI_GRAYD, "font": "Arial 9 bold"}
SLEEP_SCALE = {"from_": 2, "to": 100, "width": 10, "resolution": 2, "border": 0,
               "highlightbackground": GUI_BG, "showvalue": False, "bg": GUI_BLUE, "orient": tk.HORIZONTAL,
               "cursor": "hand2", "troughcolor": GUI_SHADOW, "sliderrelief": tk.FLAT, "activebackground": GUI_BLUE}


class TkOptionFrame:
    def __init__(self, parent):
        self.parent = parent
        self.frame = tk.Frame(self.parent.video_miner, bg=GUI_BG, bd=10)
        self.frame.pack(side="top", fill="x", padx=0, pady=0)

        # variable to prevent pause/play toggling while editing object class
        self.changing_object = False

        # create a frame to center entry fields
        self.entry_grid = tk.Frame(self.frame)
        self.entry_grid.pack(side="top")
        tk.Grid.rowconfigure(self.entry_grid, 0, weight=1, minsize=15)
        tk.Grid.columnconfigure(self.entry_grid, 0, weight=1)

        # the frame that holds the actual fields (possibly redundant)
        self.entry_grid_fields = tk.Frame(self.entry_grid, bg=GUI_BG)
        self.entry_grid_fields.grid(row=0)
        tk.Grid.rowconfigure(self.entry_grid_fields, 0, weight=1)

        # entry for object class' name
        tk.Label(self.entry_grid_fields, LABEL_TITLE, text="Object class:").grid(column=2, row=0, pady=1)
        self.object_class = tk.StringVar()
        e1 = tk.Entry(self.entry_grid_fields, textvariable=self.object_class, fg=GUI_GRAYD, font=("Arial", 11, "bold"))
        e1.grid(column=2, row=2, sticky=tk.N)
        e1.insert(0, "default_class")
        e1.bind("<FocusIn>", self.callback_entry1_focus)
        e1.bind("<FocusOut>", self.callback_entry1_nofocus)

        # entry to edit the export interval value n
        tk.Label(self.entry_grid_fields, LABEL_TITLE, text="Export interval:").grid(column=1, row=0)
        self.n = tk.StringVar()
        e = tk.Entry(self.entry_grid_fields, textvariable=self.n, fg=GUI_GRAYD, font=("Arial", 11, "bold"))
        e.grid(column=1, row=2, padx=30, sticky=tk.N)
        e.insert(0, STORE_N)

        # option menu to select which tracking algorithm is used
        tk.Label(self.entry_grid_fields, LABEL_TITLE, text="Tracking algorithm:").grid(column=0, row=0)
        self.algorithm_selection = tk.Listbox(self.entry_grid_fields, selectmode=tk.BROWSE)
        self.algorithm_selection.configure(relief="flat", cursor="hand2", takefocus=0, activestyle='none', bd=0,
                                           height=2,
                                           highlightcolor=GUI_BG, highlightbackground=GUI_BG,
                                           font=("Arial", 11, "bold"),
                                           selectbackground=GUI_SHADOW, fg=GUI_GRAYD, listvariable=0,
                                           selectforeground=GUI_GRAYD,
                                           exportselection=False)
        self.algorithm_selection.bind('<<ListboxSelect>>', self.update_algorithm)
        # TODO change to for loop over algorithm folder/list
        self.algorithm_selection.insert(0, " Re3 (default)")
        self.algorithm_selection.insert(1, " CMT ")
        self.algorithm_selection.select_set(0)
        self.algorithm_selection.grid(column=0, row=2, sticky=tk.NSEW)

        # scale to control sleep time between two frames while playing
        self.speed_scale_string = tk.StringVar()
        self.speed_scale_string.set("Sleep time \nregular playback: ({})  ".format(self.parent.speed))
        self.speed_label = tk.Label(self.entry_grid_fields, LABEL_TITLE, textvariable=self.speed_scale_string).grid(
            column=3, row=0)
        self.speed_scale = tk.Scale(self.entry_grid_fields, SLEEP_SCALE)
        self.speed_scale.grid(button_config, column=3, row=2, padx=30)
        self.speed_scale.bind("<ButtonRelease-1>", self.update_speed)

        # scale to control sleep time between two frames while tracking
        self.track_scale_string = tk.StringVar()
        self.track_scale_string.set("Sleep time \ntracking: ({})  ".format(self.parent.track_speed))
        self.track_label = tk.Label(self.entry_grid_fields, LABEL_TITLE, textvariable=self.track_scale_string)
        self.track_label.grid(column=4, row=0)
        self.track_scale = tk.Scale(self.entry_grid_fields, SLEEP_SCALE)
        self.track_scale.grid(button_config, column=4, row=2)
        self.track_scale.bind("<ButtonRelease-1>", self.update_track_speed)

    # updates playback sleep time and label
    def update_speed(self, _):
        self.parent.speed = self.speed_scale.get()
        self.speed_scale_string.set("Sleep time \nregular playback: ({})  ".format(self.parent.speed))

    # updates tracking sleep time and label
    def update_track_speed(self, _):
        self.parent.track_speed = self.track_scale.get()
        self.track_scale_string.set("Sleep time \ntracking: ({})  ".format(self.parent.track_speed))

    # updates which tracking algorithm is used
    def update_algorithm(self, _):
        index = self.algorithm_selection.curselection()[0]
        self.parent.tracking = False
        self.parent.tracker = self.parent.tracker_list[index]
        self.parent.video_loop(True)
        self.parent.control_panel.pause_playing()

    # function to prevent spacebar from pausing while typing object class
    def callback_entry1_nofocus(self, _):
        self.changing_object = False

    # function to prevent spacebar from pausing while typing object class
    def callback_entry1_focus(self, _):
        self.changing_object = True

    # returns the value of the object class which is used to store in the dataset
    def get_object_class(self):
        string = self.object_class.get().strip().replace(" ", "_")
        self.object_class.set(string)
        return string

    # returns the interval value n which determines how often a frame is stored to the dataset
    def get_n(self):
        try:
            return int(self.n.get())
        # if it is not possible convert value to int set value to default value
        except ValueError:
            self.n.set(STORE_N)
            return STORE_N
