import threading
import datetime
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from functools import partial
from settings import GUI_BG, GUI_GRAYD, GUI_BLUE, GUI_RED, GUI_SHADOW, data_set_previewsize

FRAME_SETTINGS = {"bg": GUI_BG}
LABEL_TITLE = {"bg": GUI_BG, "fg": GUI_GRAYD, "font": "Arial 11 bold", "cursor": "hand2"}
LABEL_IMG = {"bg": GUI_BG, "cursor": "hand2"}
img_padding = 5
BUTTON_LAYOUT = {"cursor": "hand2", "pady": 5, "bd": 0, "bg": GUI_BG, "fg": GUI_GRAYD, "font": "Arial 11 bold"}

EDIT_BUTTON_LAYOUT = {"cursor": "hand2", "pady": 5, "bd": 1, "activebackground": GUI_BLUE, "bg": GUI_BLUE,
                      "activeforeground": GUI_BG, "fg": GUI_BG, "font": "Arial 9 bold"}

SCROLL_STYLE = {"bg": GUI_BG, "cursor": "hand2"}

IMAGE_BUTTON = {"bg": GUI_BG, "activebackground": GUI_BG, "cursor": "hand2", "border": 0, "relief": "solid",
                "highlightbackground": GUI_BLUE, "highlightcolor": GUI_BLUE}

threading.Lock()


class TkDatasetFrame:
    def __init__(self, parent):
        self.parent = parent
        self.parent.dataset_explorer.grid_columnconfigure(1, weight=1)
        self.parent.dataset_explorer.grid_rowconfigure(0, weight=1)

        # variables used for drawing:
        self.class_dict = self.parent.dataset_dict

        self._after_id = None
        self.frame_list = []
        self.image_list = []
        self.button_list = {}
        self.current_class = self.parent.current_object
        self.current_selection = []
        self.selected_button_ids = []
        self.prev_size = None

        # =========== class list ===============#
        self.class_frame = tk.Frame(self.parent.dataset_explorer, FRAME_SETTINGS)
        self.class_frame.grid(row=0, column=0, sticky="nsew", pady=10)

        self.class_label = tk.Label(self.class_frame, LABEL_TITLE, text="Classes:", width=25)
        self.class_label.pack(side="top", fill="x")

        spacing = tk.Frame(self.class_frame, bg=GUI_SHADOW, height=1)
        spacing.pack(side="top", fill="x", padx=10)

        # =========== images list ===============#
        self.image_frame_base = tk.Frame(self.parent.dataset_explorer, bg=GUI_BG, bd=0)
        self.image_frame_base.grid(row=0, column=1, sticky="nsew", pady=10)

        self.image_header = tk.StringVar()
        self.image_header.set("Images:")
        self.images_label = tk.Label(self.image_frame_base, LABEL_TITLE, textvariable=self.image_header)
        self.images_label.pack(side="top", fill="x")

        spacing = tk.Frame(self.image_frame_base, bg=GUI_SHADOW, height=1)
        spacing.pack(side="top", fill="x", padx=10)

        self.canvas = tk.Canvas(self.image_frame_base, bg=GUI_BG, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=1)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.image_frame = tk.Frame(self.canvas, bg=GUI_BG, bd=0)
        tk.Grid.columnconfigure(self.image_frame, 0, weight=1)

        self.interior_id = self.canvas.create_window((0, 0), window=self.image_frame, anchor='nw')

        self.img_per_row = None

        # =========== settings / export list ===============#
        self.settings_frame = tk.Frame(self.parent.dataset_explorer, bg=GUI_BG, width=200)
        self.settings_frame.grid(row=0, column=2, sticky="nsew", pady=10)

        self.settings_label = tk.Label(self.settings_frame, LABEL_TITLE, text="Edit:", width=25)
        self.settings_label.pack(side="top", fill="x")

        spacing = tk.Frame(self.settings_frame, bg=GUI_SHADOW, height=1)
        spacing.pack(side="top", fill="x", padx=10)

        self.delete_button = None
        self.destination_option = None
        self.move_button = None
        self.export_button = None

        # resize and scroll functions
        self.canvas.bind('<Configure>', self._configure_canvas)
        self.image_frame.bind('<Configure>', self._configure_interior)

    # scroll canvas on mousewheel scroll
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 100)), "units")

    # track changes to the canvas and frame width and sync them,
    # also updating the scrollbar
    def _configure_interior(self, _):
        # update the scrollbars to match the size of the inner frame
        size = (self.image_frame.winfo_reqwidth(), self.image_frame.winfo_reqheight())
        self.canvas.config(scrollregion="0 0 %s %s" % size)
        if self.image_frame.winfo_reqwidth() != self.canvas.winfo_width():
            # update the canvas's width to fit the inner frame
            self.canvas.config(width=self.image_frame.winfo_reqwidth())

    def _configure_canvas(self, _):
        if self.image_frame.winfo_reqwidth() != self.canvas.winfo_width():
            # update the inner frame's width to fill the canvas
            self.canvas.itemconfigure(self.interior_id, width=self.canvas.winfo_width())

    @staticmethod
    def clear(list_of_frames):
        for frame in list_of_frames:
            frame.destroy()

    # prepares the frames and buttons for display of images classes en buttons
    def reset_frames(self, update_current_class=True):
        self.class_dict = self.parent.dataset_dict
        self.clear(self.frame_list)
        self.canvas.yview_moveto(0)

        if update_current_class or not list(self.class_dict[self.current_class]):
            self.current_class = self.parent.current_object

        # initialise delete button if not yet done
        if not self.delete_button and self.current_class:
            self.delete_button = tk.Button(self.settings_frame, EDIT_BUTTON_LAYOUT, text='Delete selected images',
                                           command=self.delete_selected)
            self.delete_button.pack(side="top", fill="x", pady=10, padx=20)

            self.export_button = tk.Button(self.settings_frame, EDIT_BUTTON_LAYOUT, text='Export dataset',
                                           command=self.export_dataset_thread)
            self.export_button.pack(side="bottom", fill="x", pady=10, padx=40)

    # bad threading function to speedup add images:
    def add_classes_thread(self):
        object_classes = list(self.class_dict.keys())
        t = threading.Thread(target=self.add_classes(object_classes))
        t.deamon = True
        t.start()

    # adds buttons for each existing class to the class frame
    # which allow the user to display the images of that class
    def add_classes(self, object_classes):
        for key in object_classes:
            if list(self.class_dict[key]):
                total = tk.Frame(self.class_frame, bg=GUI_SHADOW, height=1)
                total.pack(side="top", fill="x", padx=10)

                title_string = key + " ({})".format(len(self.class_dict[key]))
                action_with_arg = partial(self.draw_images, key)
                title = tk.Button(total, BUTTON_LAYOUT, text=title_string, command=action_with_arg)
                title.pack(side="top", fill="x", )

                spacing = tk.Frame(total, bg=GUI_SHADOW, height=1)
                spacing.pack(side="top", fill="x", padx=10)

                self.frame_list.append(total)

        # if classes available automatically draw them
        if object_classes:
            self.draw_images(self.current_class)

    # TODO try adding images more images on scroll instead of all at once, or pagination
    # updates the image frame with the images without the use of threads
    def draw_images(self, object_class):
        self.current_class = object_class
        self.current_selection = []

        # remove currently displayed images
        self.clear(self.image_list)

        # reset scroll position to top and set header to fit current class
        self.image_header.set(str(object_class) + " images:")
        self.canvas.yview_moveto(0)
        width = self.canvas.winfo_width()
        ipad = 2

        # calculate the number of images that fit a row given the current screen width
        img_per_row = int((width - 20) / (data_set_previewsize + 2 * (ipad + img_padding)))
        self.img_per_row = img_per_row

        self.button_list = {}

        row_num = 0
        col_num = 0
        row = tk.Frame(self.image_frame, FRAME_SETTINGS)

        # draw the images buttons on row frames which are added to the main image frame
        for img_id, img_list in self.class_dict[object_class].items():
            [_, padded_photo] = img_list

            image_button = tk.Button(row, IMAGE_BUTTON, image=padded_photo,
                                     command=lambda name=img_id: self.add_to_selected(name))
            image_button.image = padded_photo
            image_button.grid(column=col_num, row=0, ipady=ipad, ipadx=ipad, sticky="nsew",
                              padx=img_padding, pady=img_padding)
            self.button_list[img_id] = image_button
            col_num += 1
            if col_num == img_per_row:
                row.grid(row=row_num, column=0)
                self.image_list.append(row)
                row = tk.Frame(self.image_frame, FRAME_SETTINGS)
                row_num += 1
                col_num = 0

        row.grid(row=row_num, column=0)
        self.image_list.append(row)

    # use thread to prevent gui form freezing when exporting dataset
    def export_dataset_thread(self):
        t = threading.Thread(target=self.export_dataset)
        t.deamon = True
        t.start()

    # export dataset, store images in user selected folder
    def export_dataset(self):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/datasets"
        folder_selected = filedialog.askdirectory(initialdir=dir_path)
        time = datetime.datetime.now().strftime("%H_%M")
        base_folder = "/Dataset_" + time + "/"

        directory = folder_selected + base_folder
        if not os.path.exists(directory):
            os.makedirs(directory)
        for folder in self.class_dict.keys():
            sub_dir = directory + "/" + folder
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            # copy to prevent changing dictionary errors while exporting and tracking at the same time
            copy_dict = self.class_dict[folder].copy()
            for image_id in copy_dict:
                filename = sub_dir + "/" + str(folder) + "_" + str(image_id) + ".jpg"
                # open image and convert back to rgb
                img = copy_dict[image_id][0][..., ::-1]

                cv2.imwrite(filename, img)
        self.parent.status_bar.set("Successfully exported dataset.")

    # deletes all selected items from the dataset
    def delete_selected(self):
        num = len(self.current_selection)
        # check if there are images to delete
        if num > 0:
            for item in self.current_selection:
                del self.class_dict[self.current_class][item]
            if self.current_class:
                self.reset_frames(False)
                self.add_classes_thread()

    # TODO: implement moving selected images to different class
    # move selected images to a different object class
    def move_selected(self):
        print("moving", self.current_selection)

    # adds an image to the list of the currently selected images to be able to delete or move a selection of images
    def add_to_selected(self, count_id):
        button = self.button_list[count_id]
        if button["bg"] == GUI_BG:
            self.current_selection.append(count_id)
            button.config(bg=GUI_RED, activebackground=GUI_RED)
            self.selected_button_ids.append(count_id)
        else:
            self.current_selection.remove(count_id)
            button.config(bg=GUI_BG, activebackground=GUI_BG)
            self.selected_button_ids.remove(count_id)
