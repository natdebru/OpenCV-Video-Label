import os
import cv2
import mss
import urllib.request
import numpy as np
from PIL import Image, ImageTk, ImageGrab
from tkinter import simpledialog, filedialog
from settings import SUPPORTED_FILES, VIDEO_H, VIDEO_W


# allows the playback of regular video files
class VideoStream:
    def __init__(self, parent):
        self.parent = parent
        self.source = "video"
        self.sleep = 1
        self.stream = None
        self.resize = None
        self.init()

    # make sure the file is supported and prepare for playback
    def init(self):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/video"
        filename = filedialog.askopenfilename(title='Open a video', initialdir=dir_path)
        filetype = filename.split(".")[-1]
        if filename:
            if filetype in SUPPORTED_FILES:
                self.stream = cv2.VideoCapture(filename)
                self.parent.frame_count = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
                self.parent.control_panel.scale["to"] = self.parent.frame_count
                self.parent.control_panel.scale.set(0)
                self.parent.status_bar.set('Opening: %s' % (filename.split("/")[-1]))
            else:
                self.parent.status_bar.set('Unsupported filetype: .{}'.format(filetype))
        else:
            self.parent.status_bar.set('No file selected.')

    # calculates the new resize values to maintain aspect ratio when resizes input images
    def calc_resize(self, input_image, max_w=VIDEO_W, max_h=VIDEO_H):
        (h, w) = input_image.shape[:2]
        if 0 not in [h, w]:
            resize_x = max_w / w
            resize_y = max_h / h
            if resize_y < resize_x:
                self.resize = tuple((int(w * resize_y), max_h))
            else:
                self.resize = tuple((max_w, int(h * resize_y)))

    # returns the next frame from the video file
    def read(self):
        available, new_frame = self.stream.read()
        if available:
            if not self.resize:
                self.calc_resize(new_frame)
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            new_frame = cv2.resize(new_frame, self.resize, interpolation=cv2.INTER_LINEAR)
        return available, new_frame

    # sets the location of the video to a given frame number
    def set(self, frame_num):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # returns the current frame number
    def get(self, _):
        return self.stream.get(cv2.CAP_PROP_POS_FRAMES)

    # release the video
    def release(self):
        self.stream.release()


# allows playback of webcam streaming
class WebcamStream:
    def __init__(self, parent):
        self.parent = parent
        self.source = "webcam"
        self.sleep = 1
        self.stream = cv2.VideoCapture(0)
        self.resize = None
        self.init()

    # prepare gui
    def init(self):
        self.parent.status_bar.set('Starting ' + self.source + ' stream...')
        self.parent.frame_count = 0
        self.parent.control_panel.scale["to"] = self.parent.frame_count

    # calculates the new resize values to maintain aspect ratio when resizes input images
    def calc_resize(self, input_image, max_w=VIDEO_W, max_h=VIDEO_H):
        (h, w) = input_image.shape[:2]
        if 0 not in [h, w]:
            resize_x = max_w / w
            resize_y = max_h / h
            if resize_y < resize_x:
                self.resize = tuple((int(w * resize_y), max_h))
            else:
                self.resize = tuple((max_w, int(h * resize_y)))

    # read prepare and return a new frame from the webcam
    def read(self):
        new_frame_available, new_frame = self.stream.read()
        if new_frame_available:
            if not self.resize:
                self.calc_resize(new_frame)
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            new_frame = cv2.resize(new_frame, self.resize)
        return new_frame_available, cv2.flip(new_frame, 1)

    # set webcam to 0, function to prevent errors
    def set(self, _):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # return 0, function to prevent errors
    def get(self, _):
        return 0

    # release webcam
    def release(self):
        self.stream.release()


# allows playback of mobilewebcam streams
class IpWebcamStream:
    def __init__(self, parent):
        self.parent = parent
        self.source = "ipwebcam"
        self.sleep = 1
        self.stream = None
        self.resize = None
        self.init()

    # prompt to obtain streams ip
    def init(self):
        self.parent.first_frame = True
        self.stream = str(
            simpledialog.askstring("Enter the IP address of your stream", "Enter the IP address of your stream:",
                                   initialvalue="http://192._._._:8080")) + "/shot.jpg"

    # calculates the new resize values to maintain aspect ratio when resizes input images
    def calc_resize(self, input_image, max_w=VIDEO_W, max_h=VIDEO_H):
        (h, w) = input_image.shape[:2]
        if 0 not in [h, w]:
            resize_x = max_w / w
            resize_y = max_h / h
            if resize_y < resize_x:
                self.resize = tuple((int(w * resize_y), max_h))
            else:
                self.resize = tuple((max_w, int(h * resize_y)))

    # read the next frame and return
    def read(self):
        try:
            self.parent.status_bar.set("Connecting with IP Webcam")
            with urllib.request.urlopen(self.stream) as imgResp:
                if imgResp:
                    new_frame = cv2.imdecode(np.array(bytearray(imgResp.read()), dtype=np.uint8), -1)
                    if not self.resize:
                        self.calc_resize(new_frame)
                    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                    new_frame = cv2.resize(new_frame, self.resize)
                    new_frame = cv2.flip(new_frame, 1)
                    return True, new_frame
        except Exception as e:
            self.parent.control_panel.stop()
            self.parent.tracking = False
            self.parent.status_bar.set("Connection with IP Webcam failed: " + str(e))
            return False, None

    # set webcam to 0, function to prevent errors
    def set(self, _):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # return 0, function to prevent errors
    def get(self, _):
        return 0

    # function to prevent errors
    def release(self):
        return


# screencapture
sct = mss.mss()


# allows user to use screen recording as video input
class ScreenCapture:
    def __init__(self, parent):
        self.parent = parent
        self.source = "screencapture"
        self.sleep = 1
        self.stream = None
        self.screen_region = None
        self.resize = (VIDEO_W, VIDEO_H)
        self.init()

    # display image on the gui
    def send_image_to_screen(self, image):
        self.parent.frame = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.parent.video_frame.frame_image["image"] = self.parent.frame
        self.parent.video_frame.frame_image.photo = self.parent.frame

    # prepare screen region capturing
    def init(self):
        self.parent.status_bar.set('Starting screen capture')

        # grab the current image from the screen
        new_frame = cv2.resize(np.array(ImageGrab.grab()), self.resize, interpolation=cv2.INTER_LINEAR)

        # display the current image on the gui to allow region selection
        self.send_image_to_screen(new_frame)
        self.get_region_of_screen()

    # let user select region of screen to watch
    def get_region_of_screen(self):
        self.parent.video_frame.get_rect()
        self.stream = True

    # calculates the new resize values, to maintain aspect ratio when resizing input images
    def calc_resize(self, input_image, max_w=VIDEO_W, max_h=VIDEO_H):
        (h, w) = input_image.shape[:2]
        if 0 not in [h, w]:
            resize_x = max_w / w
            resize_y = max_h / h
            if resize_y < resize_x:
                self.resize = tuple((int(w * resize_y), max_h))
            else:
                self.resize = tuple((max_w, int(h * resize_y)))

    # convert cropped coordinates to real screen coordinates
    def calc_region_of_screen(self):
        width = self.parent.roi_br[0] - self.parent.roi_tl[0]
        height = self.parent.roi_br[1] - self.parent.roi_tl[1]

        monitor_width = self.parent.root.winfo_screenwidth()
        monitor_height = self.parent.root.winfo_screenheight()

        resizex = monitor_width / self.resize[0]
        resizey = monitor_height / self.resize[1]
        tlx, tly = int(self.parent.roi_tl[0] * resizex), int(self.parent.roi_tl[1] * resizey)
        new_width = int(width * resizex)
        new_height = int(height * resizey)
        self.screen_region = {'top': tly, 'left': tlx, 'width': new_width, "height": new_height}

    # read and return next frame
    def read(self):
        if not self.screen_region:
            self.calc_region_of_screen()
            self.resize = None

        res = sct.grab(self.screen_region)
        new_frame = np.array(Image.frombytes("RGB", res.size, res.rgb, "raw", "BGR"))

        if not self.resize:
            self.calc_resize(new_frame)

        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        new_frame = cv2.resize(new_frame, self.resize)
        return True, new_frame

    # function to prevent errors
    def set(self, _):
        return

    # function to prevent errors
    def get(self, _):
        return 0

    # function to prevent errors
    def release(self):
        return
