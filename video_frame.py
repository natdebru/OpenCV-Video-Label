import cv2
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from constants import VIDEO_BACKGROUND_IMG, VIDEO_H, VIDEO_W, VIDEO_PATH, GUI_RED


# frame which holds the video frames
class TkVideoFrame:
    def __init__(self, parent):

        # settings:
        self.video_size = (VIDEO_W, VIDEO_H)
        self.roi = [[0, 0], [0, 0]]
        self.parent = parent
        self.root = parent.root
        self.default_image = VIDEO_BACKGROUND_IMG

        # the main frame:
        self.rect_canvas = None
        self.frame = tk.Frame(self.parent.video_miner, bg="black", bd=0)
        self.frame.pack(fill="both")
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # initialize the video display with an image
        self.frame_image = tk.Label(self.frame, bd=0, highlightthickness=0, relief='flat')
        self.set_image(self.default_image)
        self.frame_image.grid(row=0, column=0)

    # display an image on the video frame
    def set_image(self, image):
        new_frame = cv2.imread(image, 3)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        self.parent.cur_image = cv2.resize(new_frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)
        self.parent.prev_image = self.parent.cur_image

        self.parent.frame = Image.fromarray(self.parent.cur_image)
        self.parent.frame = ImageTk.PhotoImage(image=self.parent.frame)

        self.frame_image["image"] = self.parent.frame
        self.frame_image.photo = self.parent.frame

    # determines what happens after tracking button is pressed
    def get_rect(self):
        # if no video, do nothing
        if not self.parent.video:
            return
        # if tracker not ready, do nothing and warn user (fixed with load screen)
        elif not self.parent.tracker and self.parent.video.source != "screencapture":
            self.parent.status_bar.set("Please wait while re3 tracker is loading...")
        # if selecting region of interest, cancel region selection and continue playing
        elif self.parent.selecting_roi:
            self.parent.selecting_roi = False
            self.parent.tracking = False
            self.rect_canvas.canvas.destroy()
            self.rect_canvas = None
            if self.parent.solitaire_mode:
                self.parent.solitaire_video.release()
            self.parent.control_panel.start_playing()
        # if already tracking, stop tracking
        elif self.parent.tracking:
            if self.parent.solitaire_mode:
                self.parent.solitaire_video.release()
            self.parent.tracking = False
            self.parent.status_bar.set("Stopped tracking.")
            self.parent.solitaire_image = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
        # if not yet doing anything, let user select an object of interest
        elif (self.parent.frame and not self.parent.selecting_roi and self.parent.video.source != "screencapture") or (
                self.parent.video.source == "screencapture" and self.parent.video.screen_region):
            if self.parent.solitaire_mode:
                self.parent.solitaire_video = cv2.VideoWriter(VIDEO_PATH + 'solitair_video.avi',
                                                              cv2.VideoWriter_fourcc(*'XVID'), 24,
                                                              (VIDEO_W, VIDEO_H))
            self.parent.status_bar.set("Select your object of interest to start tracking.")
            self.parent.control_panel.pause_playing()
            self.parent.selecting_roi = True
            self.parent.tracking = True
            self.rect_canvas = RectCanvas(self.frame, self.parent)
        # else (if screencapturing), let user select a specific region of the screen to watch
        else:
            self.rect_canvas = RectCanvas(self.frame, self.parent)
            self.parent.control_panel.pause_playing()
            self.parent.status_bar.set("Select the region of the screen you want to watch.")
            self.parent.selecting_roi = True


# the canvas which allows the use of RectTracker
class RectCanvas:
    def __init__(self, master, root):
        self.root = root
        # not using VIDEO_H and VIDEO_W because of variable height and width when in screencapture
        (width, height) = self.root.video.resize
        self.canvas = tk.Canvas(master, height=height, width=width, bd=0,
                                highlightthickness=0, relief='flat', cursor="hand2")
        self.canvas.grid(row=0, column=0)
        self.canvas.photo = self.root.frame
        self.canvas.create_image(0, 0, image=self.root.frame, anchor=tk.NW)

        self.rect = RectTracker(self.canvas, self.root)
        x, y = None, None

        # draws dashed lines to indicate x and y of potential box
        def cool_design(event):
            global x, y
            self.canvas.delete('no')
            dashes = [1, 5]
            # vertical dashes
            x = self.canvas.create_line(event.x, 0, event.x, VIDEO_H, dash=dashes, tags='no')
            # horizontal dashes
            y = self.canvas.create_line(0, event.y, VIDEO_W, event.y, dash=dashes, tags='no')

        self.canvas.bind('<Motion>', cool_design, '+')
        self.rect.autodraw(fill="", width=3)


# the rectangle roi selection class
class RectTracker:
    def __init__(self, canvas, root):
        self.root = root
        self.canvas = canvas
        self.item = None
        self.start = None
        self._command = None
        self.rectopts = None

    # draw the rectangle
    def draw(self, start, end, **opts):
        return self.canvas.create_rectangle(*(list(start) + list(end)), **opts, outline=GUI_RED)

    # setup automatic drawing; supports command option
    def autodraw(self, **opts):
        self.start = None
        self.canvas.bind("<Button-1>", self.__update, '+')
        self.canvas.bind("<B1-Motion>", self.__update, '+')
        self.canvas.bind("<ButtonRelease-1>", self.__stop, '+')

        self._command = opts.pop('command', lambda *args: None)
        self.rectopts = opts

    def __update(self, event):
        if not self.start:
            self.start = [event.x, event.y]
            return
        if self.item is not None:
            self.canvas.delete(self.item)
        self.item = self.draw(self.start, (event.x, event.y), **self.rectopts)
        self._command(self.start, (event.x, event.y))

    def __stop(self, event):

        # make sure roi is [top left, bot right]
        tl_x = max(min(self.start[0], event.x), 0)
        br_x = max(self.start[0], event.x)
        tl_y = max(min(self.start[1], event.y), 0)
        br_y = max(self.start[1], event.y)

        # prevent zero size roi
        if br_y - tl_y == 0 or br_x - tl_x == 0:
            return

        self.root.roi_tl = [tl_x, tl_y]
        self.root.roi_br = [br_x, br_y]
        self.root.roi = cv2.cvtColor(self.root.cur_image[tl_y:br_y, tl_x:br_x], cv2.COLOR_BGR2RGB)
        self.root.selecting_roi = False
        self.canvas.destroy()
        self.root.status_bar.clear()

        # set the object class to value of the entry field
        self.root.current_object = self.root.tracking_options.get_object_class()
        if self.root.current_object not in self.root.dataset_dict:
            self.root.dataset_dict[self.root.current_object] = {}

        try:
            self.root.tracker.track(self.root.current_object, self.root.cur_image[:, :, ::-1], [tl_x, tl_y, br_x, br_y])
        except Exception as e:
            print("failed initializing:", e)
            self.root.tracking = False

        self.root.control_panel.start_playing()
