import cv2
import os
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from os.path import isfile, join
from PIL import Image, ImageTk
from video_sources import ScreenCapture, IpWebcamStream, WebcamStream, VideoStream
from constants import GUI_BG, GUI_RED, IMG_TYPES


class TkTopMenu(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent.root)
        self.root = parent.root
        self.parent = parent
        self.import_thread = None

        self.file_menu = tk.Menu(self, tearoff=False)
        self.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open video", command=self.open_file, accelerator="Ctrl+O")
        self.file_menu.add_command(label="Open webcam", command=self.open_webcam,
                                   accelerator="Ctrl+W")
        self.file_menu.add_command(label="Open ipwebcam", command=self.open_ipwebcam,
                                   accelerator="Ctrl+I")
        self.file_menu.add_command(label="Open screencapture", command=self.open_screencapture, accelerator="Ctrl+S")
        self.file_menu.add_command(label="Import dataset", command=self.import_dataset_thread, accelerator="Ctrl+D")

        # self.file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        self.file_menu.add_command(label="Quit", command=self.quit_app, accelerator="Ctrl+Q")

        self.help_menu = tk.Menu(self, tearoff=False)
        self.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Help", command=self.show_help, underline=1, accelerator="Ctrl+H")
        self.help_menu.add_command(label="About", command=self.about_app, underline=1, accelerator="Ctrl+A")

        # add shortcuts to the application
        self.root.bind('<Control-d>', lambda e: self.import_dataset_thread())
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-w>', lambda e: self.open_webcam())
        self.root.bind('<Control-i>', lambda e: self.open_ipwebcam())
        self.root.bind('<Control-s>', lambda e: self.open_screencapture())
        self.root.bind('<Control-q>', lambda e: self.quit_app())
        self.root.bind('<Control-h>', lambda e: self.show_help())
        self.root.bind('<Control-a>', lambda e: self.about_app())
        self.root.bind('<Control-t>', lambda e: self.parent.video_frame.get_rect())
        self.root.bind("<space>", lambda e: self.parent.control_panel.space_playpause())

    # use thread to prevent gui from freezing when importing dataset
    def import_dataset_thread(self):
        self.import_thread = threading.Thread(target=self.import_dataset)
        self.import_thread.deamon = True
        self.import_thread.start()

    # allows user to import compatible datasets
    def import_dataset(self):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/datasets"
        self.parent.gui_style.configure("red.Horizontal.TProgressbar", foreground=GUI_BG, background=GUI_RED)
        dataset_root = filedialog.askdirectory(title="Import dataset", initialdir=dir_path) + "/"
        dataset_name = dataset_root.split("/")[-2]
        subdirs = [x[0] for x in os.walk(dataset_root)]

        # check if subdirectories
        if len(subdirs) > 1:
            subdirs = subdirs[1:]
            classes = {x.split("/")[-1]: x for x in subdirs}
            total_images = sum([len(x[2]) for x in os.walk(dataset_root)][1:])
        else:
            classes = {x.split("/")[-2]: x for x in subdirs}
            total_images = sum([len(x[2]) for x in os.walk(dataset_root)])

        img_id = 0

        # popup window to display loading progress
        import_window = tk.Toplevel(self.root, bg=GUI_BG)
        import_window.attributes('-disabled', True)
        import_window.tk.call('wm', 'iconphoto', import_window._w, self.parent.icon)
        import_window.geometry("300x100")
        import_window.title('Loading ' + dataset_name + "...")
        self.center_on_screen(import_window)

        # add progressbar
        progress = ttk.Progressbar(import_window, style="red.Horizontal.TProgressbar", length=400, mode="determinate")
        progress.pack(padx=10, pady=(10,))
        progress["value"] = 0
        progress["maximum"] = total_images

        # label to display percentage progress
        percentage = tk.StringVar()
        progress_label = tk.Label(import_window, bg=GUI_BG, textvariable=percentage)
        progress_label.pack()

        # label to display which file is being processed
        file = tk.StringVar()
        file_label = tk.Label(import_window, bg=GUI_BG, textvariable=file)
        file_label.pack(side="left", padx=10)
        self.parent.dataset_dict = dict()

        # loading the files:
        for c in classes.keys():
            self.parent.dataset_dict[c] = {}
            self.parent.current_object = c

            images = [join(classes[c], f) for f in os.listdir(classes[c]) if isfile(join(classes[c], f))]
            for image in images:
                file_type = str(image.split(".")[-1])
                file.set("Loading: " + c + "_" + str(img_id) + "." + file_type)
                progress["value"] += 1
                percentage.set(str(int(progress["value"] / progress["maximum"] * 100)) + "%")
                if file_type in IMG_TYPES:
                    cropped = cv2.imread(image, 3)[..., ::-1]
                    success, padded_preview = self.parent.pad_image(cropped)
                    if success:
                        photo = Image.fromarray(padded_preview)
                        padded_preview = ImageTk.PhotoImage(image=photo)
                        self.parent.dataset_dict[c][img_id] = [cropped, padded_preview]
                        img_id += 1
                else:
                    self.parent.status_bar.set("Failed loading: " + image)

        self.parent.status_bar.set("Successfully loaded " + dataset_name)
        self.import_thread = None
        import_window.destroy()

    # open and load a video file
    def open_file(self):
        if self.parent.video:
            self.parent.control_panel.stop()
        self.parent.video = VideoStream(self.parent)
        if self.parent.video.stream:
            self.parent.control_panel.start_playing()
        else:
            self.parent.video = None

    # open the webcam if available
    def open_webcam(self):
        if self.parent.video:
            self.parent.control_panel.stop()
        self.parent.video = WebcamStream(self.parent)
        if self.parent.video.stream:
            self.parent.control_panel.start_playing()
        else:
            self.parent.video = None

    # initialise ipwebcam
    def open_ipwebcam(self):
        if self.parent.video:
            self.parent.control_panel.stop()
        self.parent.video = IpWebcamStream(self.parent)
        if self.parent.video.stream:
            self.parent.control_panel.start_playing()
        else:
            self.parent.video = None

    # allow the user to make screen capture videos
    def open_screencapture(self):
        if self.parent.video:
            self.parent.control_panel.stop()
        self.parent.video = ScreenCapture(self.parent)
        self.parent.video.get_region_of_screen()
        if self.parent.video.stream:
            self.parent.control_panel.start_playing()
        else:
            self.parent.video = None

    # store a file, (currently not doing anything)
    def save_file(self):
        filename = filedialog.asksaveasfilename()
        if filename:
            self.parent.status_bar.set('Save something to %s' % filename)

    # exit the application
    def quit_app(self):
        self.parent.quiting = True
        self.parent.status_bar.set("Closing application")
        self.root.destroy()

    # just display the help message as help is currently not available
    def show_help(self):
        self.parent.status_bar.set('Not available.')

    # display the (list of) developer(s)
    def about_app(self):
        about_dialog = tk.Toplevel(self.root, bg=GUI_BG)
        self.center_on_screen(about_dialog)
        about_dialog.tk.call('wm', 'iconphoto', about_dialog._w, self.parent.icon)
        about_dialog.minsize(250, 200)
        about_dialog.geometry("250x200")
        about_dialog.title('About ' + self.root.title())
        about_dialog.bind('<Escape>', lambda event: about_dialog.destroy())
        opencv_text = self.root.title() + " is created by: \n- Nathan de Bruijn  (natdebru)"
        re3_text = "\n\nRe3 is created by: \n- Daniel Gordon  (danielgordon10)\n- Ali Farhadi\n- Dieter Fox"
        cmt_text = "\n\nCMT is created by: \n- Georg Nebehay  (gnebehay)\n- Roman Pflugfelder"
        message_text = opencv_text + re3_text + cmt_text
        tk.Message(about_dialog, text=message_text, bg=GUI_BG, width=300).pack(fill="x", side="top", padx=10, pady=10)

    # function to center popup windows
    def center_on_screen(self, toplevel):
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w, root_h = tuple(int(_) for _ in self.root.geometry().split('+')[0].split('x'))

        toplevel.update_idletasks()
        popup_w, popup_h = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))

        x = root_x + (root_w / 2) - (popup_w / 2)
        y = root_y + (root_h / 2) - (popup_h / 2)
        toplevel.geometry('%dx%d+%d+%d' % (popup_w, popup_h, x, y))
