import cv2
import time
import top_menu
import video_frame
import dataset_frame
import control_frame
import option_frame
import status_bar
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from settings import ICON, VIDEO_H, VIDEO_W, GUI_BG, GUI_GRAYL, GUI_RED, GUI_REDD_RGB, data_set_previewsize


class MainWindow:
    def __init__(self):
        # player settings
        self.video = None
        self.frame = None
        self.cur_image = None
        self.play = False
        self.selecting_roi = False
        self.tracking = False
        self.scale_drag = False
        self.frame_count = 1000

        # variables used for screencapture region selection
        self.roi_tl = None
        self.roi_br = None

        # tracking algorithm, set to None because the used trackers are imported later on
        self.tracker = None
        self.tracker_list = []

        # dataset settings
        self.current_object = ""
        self.max_image_count = 0
        self.dataset_dict = {}

        # variables keeping sleep time values after each frame, lower is faster
        self.speed = 10
        self.track_speed = 4

        # initialize the main window
        self.root = tk.Tk()
        self.root.withdraw()
        x = int(self.root.winfo_screenwidth() / 2 - (VIDEO_W + 200) / 2)
        y = int(self.root.winfo_screenheight() / 2 - (VIDEO_H + 260) / 2)
        self.root.geometry('%dx%d+%d+%d' % (VIDEO_W + 200, VIDEO_H + 260, x, y))

        self.root.minsize(VIDEO_W, VIDEO_H + 260)
        self.root.title("OpenCV Video Label")
        self.root.wm_iconbitmap(ICON)

        # show loading screen while tracking algorithms are loaded
        loading = LoadingScreen(self.root)
        self.load_trackers()
        loading.fade_away()

        # add the top menu
        self.top_menu = top_menu.TkTopMenu(self)
        self.root.config(menu=self.top_menu, bg=GUI_BG)

        # add the status bar to the bottom of the window
        self.status_bar = status_bar.TkStatusBar(self)
        self.status_bar.pack(side="bottom", fill="x")

        # creating notebook for each app function tab
        self.gui_style = ttk.Style()
        self.gui_style.configure('My.TNotebook', background=GUI_BG, bd=0)
        self.gui_style.configure('My.TFrame', background=GUI_BG)
        self.note = ttk.Notebook(self.root, style='TNotebook')
        self.note.bind('<<NotebookTabChanged>>', self.tab_changer)

        # ================ video player tab =========================== #
        self.video_miner = ttk.Frame(self.note, style='My.TFrame')
        self.note.add(self.video_miner, text='Video miner')
        self.note.pack(expand=1, fill="both")

        # add the video frame
        self.video_frame = video_frame.TkVideoFrame(self)

        # gray border line
        spacing = tk.Frame(self.video_miner, bg=GUI_GRAYL, height=1)
        spacing.pack(fill="x", side="top", pady=0)

        # add the video control panel
        self.control_panel = control_frame.TkControlFrame(self)
        self.control_panel.pack(fill="x", side="top", padx=10, pady=10)

        # gray border line
        spacing = tk.Frame(self.video_miner, bg=GUI_GRAYL, height=1)
        spacing.pack(fill="x", side="top", pady=0, padx=10)

        # add tracking settings
        self.tracking_options = option_frame.TkOptionFrame(self)

        # ================ dataset explorer tab ========================= #
        self.dataset_explorer = ttk.Frame(self.note, style='My.TFrame')
        self.note.add(self.dataset_explorer, text='Dataset explorer')
        self.note.pack(expand=1, fill="both")

        self.dataset_frame = dataset_frame.TkDatasetFrame(self)

        # ================ init the video ============================= #
        self.quiting = False
        self.root.protocol("WM_DELETE_WINDOW", self.top_menu.quit_app)

        self.solitaire_mode = False
        self.solitaire_video = None
        self.solitaire_image = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)

        self.active_tab = "video_miner"
        self.frame_time_list = []
        self.frame_counter = 0
        self.video_loop()

    # handle change between the tabs
    def tab_changer(self, event):
        index = event.widget.index("current")
        if index == 0:
            self.active_tab = "video_miner"
        elif index == 1:
            self.control_panel.pause_playing()
            self.active_tab = "dataset_explorer"
            self.dataset_frame.reset_frames()
            self.dataset_frame.add_classes_thread()

    # import trackers and set re3 as default tracking algorithm
    def load_trackers(self):
        from algorithms.CMT import CMT
        from algorithms.re3 import re3_tracker
        self.tracker_list = [re3_tracker.Re3Tracker(), CMT.CMT(self)]
        self.tracker = self.tracker_list[0]

    # draws a red border and the label of the class on the current frame
    def draw_roi(self, tl_x, tl_y, br_x, br_y):
        font = cv2.FONT_HERSHEY_DUPLEX

        # draw the bounding box
        cv2.rectangle(self.cur_image, (tl_x, tl_y), (br_x, br_y), GUI_REDD_RGB, 2)

        # draw the class label background and label
        cv2.rectangle(self.cur_image, (tl_x - 1, tl_y - 15), (tl_x + 10 + len(self.current_object) * 10, tl_y),
                      GUI_REDD_RGB, cv2.FILLED)
        cv2.putText(self.cur_image, self.current_object, (tl_x + 5, tl_y - 2), font, .5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # creates the solitaire-like video while tracking
    def solitaire_maker(self, tl_x, tl_y, br_x, br_y):
        self.solitaire_image[tl_y:br_y, tl_x:br_x] = self.cur_image[tl_y:br_y, tl_x:br_x]
        self.solitaire_video.write(self.solitaire_image[:, :, ::-1])

    # takes an image and pads it with black borders, maintaining its original aspect ratio
    def pad_image(self, cropped, max_w=data_set_previewsize, max_h=data_set_previewsize):
        (h, w) = cropped.shape[:2]

        # prevent division by 0 on next lines and resize error
        if w > 5 and h > 5:
            resize_x = max_w / w
            resize_y = max_h / h
            top, bot, left, right = 0, 0, 0, 0

            if resize_y < resize_x:
                new_width = int(w * resize_y)
                resize = tuple((new_width, max_h))
                total_pad = max_w - new_width
                left = int(total_pad / 2)
                right = total_pad - left
            else:
                new_height = int(h * resize_x)
                resize = tuple((max_w, new_height))
                total_pad = max_h - new_height
                top = int(total_pad / 2)
                bot = total_pad - top

            cropped = cv2.resize(cropped, resize, interpolation=cv2.INTER_LINEAR)
            cropped = cv2.copyMakeBorder(cropped, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            return True, cropped

        # stop tracking if the roi's width or height is to small
        else:
            self.tracking = False
            return False, None

    # function to calculate and display the fps on gui
    def calc_fps(self):
        current_time = time.time()
        self.frame_time_list.append(current_time)

        # update interval in seconds
        update_interval = 2

        # not super precise but good enough because fps is integer
        if int(self.frame_time_list[-1] - self.frame_time_list[0]) >= update_interval:
            fps = int(len(self.frame_time_list) / update_interval)
            self.frame_time_list = []
            self.status_bar.set("FPS: " + str(fps))

    # video playing function from where tracking algorithms are called
    def video_loop(self, single_frame=False):
        [tl_x, tl_y, br_x, br_y] = [0, 0, 0, 0]
        if self.play and self.video:
            self.calc_fps()

            new_frame_available, self.cur_image = self.video.read()

            if new_frame_available:
                self.frame_counter += 1

                # update the video location scale to location of the new frame number
                if not self.control_panel.scale_drag:
                    self.control_panel.scale.set(self.video.get(cv2.CAP_PROP_POS_FRAMES))

                # use tracker to locate the object in the new frame
                if self.tracking:
                    try:
                        [tl_x, tl_y, br_x, br_y] = self.tracker.track(self.current_object, self.cur_image[:, :, ::-1])
                    # except to catch cmt bugs
                    except Exception as e:
                        print(e)
                        self.tracking = False

                    # store image to dataset
                    if self.frame_counter % self.tracking_options.get_n() == 0:
                        cropped_roi = self.cur_image[tl_y:br_y, tl_x:br_x].copy()
                        success, padded_preview = self.pad_image(cropped_roi)
                        if success:
                            photo = Image.fromarray(padded_preview)
                            padded_preview = ImageTk.PhotoImage(image=photo)
                            self.dataset_dict[self.current_object][self.max_image_count] = [cropped_roi, padded_preview]
                            self.max_image_count += 1
                    self.draw_roi(tl_x, tl_y, br_x, br_y)

                # display the current frame on the gui
                if self.tracking and self.solitaire_mode:
                    self.solitaire_maker(tl_x, tl_y, br_x, br_y)
                    self.frame = Image.fromarray(self.solitaire_image)
                else:
                    self.frame = Image.fromarray(self.cur_image)

                self.frame = ImageTk.PhotoImage(image=self.frame)
                self.video_frame.frame_image["image"] = self.frame
                self.video_frame.frame_image.photo = self.frame

        # handle sleep time between frames depending on whether user is tracking
        if not single_frame:
            if self.tracking:
                self.root.after(self.track_speed, self.video_loop)
            else:
                self.root.after(self.speed, self.video_loop)


# loading screen
class LoadingScreen(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.overrideredirect(1)
        self.configure(background=GUI_RED)

        self.width = 300
        self.height = 300
        self.border_width = 23
        self.font_size = 16

        # center the window on the users screen
        x = int(self.winfo_screenwidth() / 2 - self.width / 2) + 5
        y = int(self.winfo_screenheight() / 2 - self.height / 2) - 125
        self.geometry('%dx%d+%d+%d' % (self.width, self.height, x, y))

        loadframe = tk.Frame(self, bg=GUI_BG)
        opencv_text = tk.Label(loadframe, text=parent.title(), font=("Arial", self.font_size, "bold"), bg=GUI_BG)
        copyright_text = tk.Label(loadframe, text="Made by Nathan de Bruijn", font=("Arial", 7), bg=GUI_BG)

        loadframe.pack(side="top", padx=self.border_width, pady=self.border_width, fill="both")
        opencv_text.pack(side="top", pady=int((self.height - self.font_size - 2 * self.border_width - 30) / 2))
        copyright_text.pack(side="bottom")
        self.update()

    # fade away animation after the imports are complete
    def fade_away(self):
        alpha = self.attributes("-alpha")
        if alpha > 0:
            alpha -= .01
            self.attributes("-alpha", alpha)
            self.after(10, self.fade_away)
        else:
            self.parent.deiconify()
            self.destroy()


if __name__ == "__main__":
    app = MainWindow()
    app.root.mainloop()
