import tooltip
import tkinter as tk
import itertools as it
from PIL import Image, ImageTk
from constants import FRAME_SETTINGS, GUI_BG, GUI_BLUE, WINDOWS_GRAYL, BUTTON_SETTINGS, button_config


# frame which hold the buttons to control video playback
class TkControlFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent.video_miner, FRAME_SETTINGS)
        self.parent = parent
        self.video_frame = parent.video_frame

        self.scale_drag = False
        tk.Grid.rowconfigure(self, 0, weight=1, minsize=15)
        tk.Grid.columnconfigure(self, 0, weight=1)

        # button images:
        self.decrease_photo = tk.PhotoImage(file="bitmaps/rewind.png")
        self.increase_photo = tk.PhotoImage(file="bitmaps/fast_forward.png")
        self.pause_photo = tk.PhotoImage(file="bitmaps/pause.png")
        self.play_photo = tk.PhotoImage(file="bitmaps/play.png")
        self.stop_photo = tk.PhotoImage(file="bitmaps/stop.png")
        self.track_photo = tk.PhotoImage(file="bitmaps/track.png")
        self.play_cycle = it.cycle([self.pause_photo, self.play_photo])

        # using a tkinter scale to control current location in video
        self.scale = tk.Scale(self, from_=0, to=self.parent.frame_count, resolution=1, width=15,
                              border=0, highlightbackground=GUI_BG, showvalue=False, orient=tk.HORIZONTAL,
                              cursor="hand2",
                              troughcolor=WINDOWS_GRAYL, sliderrelief=tk.FLAT, bg=GUI_BLUE,
                              activebackground=GUI_BLUE)
        self.scale.grid(row=0, column=0, sticky=tk.NSEW, pady=10)
        self.scale.bind("<Button-1>", self.drag_switch)
        self.scale.bind("<ButtonRelease-1>", self.update_video_location)

        # the frame that holds the control buttons
        self.button_panel = tk.Frame(self, bg=GUI_BG)
        self.button_panel.grid(columnspan=2, row=1)
        tk.Grid.rowconfigure(self.button_panel, 0, weight=1, minsize=40)

        # tracking button
        self.trackbut = tk.Button(self.button_panel, BUTTON_SETTINGS, command=self.video_frame.get_rect,
                                  image=self.track_photo)
        self.trackbut.grid(button_config, column=0)
        tooltip.CreateToolTip(self.trackbut, "Toggle tracking")

        # play pause button
        self.playbut = tk.Button(self.button_panel, BUTTON_SETTINGS, command=self.playpause, image=self.play_photo)
        self.playbut.grid(button_config, column=1)
        tooltip.CreateToolTip(self.playbut, "Play/pause")

        # playing speed decrease button
        decreasebut = tk.Button(self.button_panel, BUTTON_SETTINGS, command=self.decrease_speed,
                                image=self.decrease_photo)
        decreasebut.grid(button_config, column=2)
        tooltip.CreateToolTip(decreasebut, "Decrease speed")

        # stop button
        stopbut = tk.Button(self.button_panel, BUTTON_SETTINGS, command=self.stop)
        stopbut.config(image=self.stop_photo, height=24, width=24)
        stopbut.grid(button_config, column=3)
        tooltip.CreateToolTip(stopbut, "Stop playing")

        # playing speed increase button
        increasebut = tk.Button(self.button_panel, BUTTON_SETTINGS, command=self.increase_speed,
                                image=self.increase_photo)
        increasebut.grid(button_config, column=4)
        tooltip.CreateToolTip(increasebut, "Increase speed")

    # spacebar pause to prevent class editing problems
    def space_playpause(self):
        if not self.parent.tracking_options.changing_object:
            self.playpause()

    # toggle between playing and pausing
    def playpause(self):
        if self.parent.video:
            if self.parent.play:
                self.pause_playing()
            else:
                self.start_playing()
        else:
            self.parent.top_menu.open_file()
            if self.parent.video:
                self.playbut['image'] = next(self.play_cycle)

    # stop playing
    def stop(self):
        if self.parent.video:
            self.parent.play = True
            self.parent.tracking = False
            self.playbut['image'] = self.play_photo
            self.parent.video.release()
            self.parent.video = None
            self.parent.video_frame.set_image(self.parent.video_frame.default_image)
            self.parent.status_bar.init()
            self.parent.fps = []
            self.scale.set(0)

    # decrease playing speed of video
    def decrease_speed(self):
        print("not doing anything right now", self)

    # increase playing speed of video
    def increase_speed(self):
        print("not doing anything right now", self)

    # start playing, update button image
    def start_playing(self):
        if not self.parent.selecting_roi:
            self.playbut['image'] = self.pause_photo
            self.play_cycle = it.cycle([self.pause_photo, self.play_photo])
            self.parent.play = True

    # pause playing, update button image
    def pause_playing(self):
        if not self.parent.selecting_roi:
            self.playbut['image'] = self.play_photo
            self.play_cycle = it.cycle([self.pause_photo, self.play_photo])
            self.parent.play = False

    # make sure the scale does not reset while setting a new video location
    def drag_switch(self, _):
        self.scale_drag = True

    # function to set a new location in the video when scale is clicked
    def update_video_location(self, _):
        if self.parent.video:
            self.scale.set(self.scale.get())
            self.parent.video.set(self.scale.get())
            self.scale_drag = False

            # set the video to the new current frame
            new_frame_available, new_frame = self.parent.video.read()
            if new_frame_available:

                self.parent.prev_frame = new_frame
                self.parent.cur_frame = self.parent.prev_frame

                self.parent.frame = Image.fromarray(self.parent.cur_frame)
                self.parent.frame = ImageTk.PhotoImage(image=self.parent.frame)

                self.parent.video_frame.frame_image["image"] = self.parent.frame
                self.parent.video_frame.frame_image.photo = self.parent.frame
            else:
                self.scale.set(0)
        else:
            self.scale.set(0)
