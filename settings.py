# video size settings (larger video size means lower fps):
resize_factor = .75
VIDEO_W = int(1250 * resize_factor)
VIDEO_H = int(720 * resize_factor)

ICON = 'bitmaps/icon.ico'
VIDEO_BACKGROUND_IMG = "bitmaps/video_bg.jpg"

# color palette :
GUI_RED = "#D70027"
GUI_REDD = "#620000"
GUI_REDD_RGB = (215, 0, 39)
GUI_BLUE = "#007ad7"
GUI_BLUED = "#01243f"
GUI_ORANGE = "#ff9200"
GUI_ORANGED = "#623800"
GUI_GREEN = "#00e920"
GUI_GREEND = "#004e20"
GUI_WHITE = "#FFFFFF"
GUI_GRAYL = "#DDDDDD"
WINDOWS_GRAYL = "#EDEDED"
GUI_GRAYD = "#272727"
GUI_BG = GUI_WHITE
GUI_SHADOW = GUI_GRAYL

# determines once how many frames an image is added to the dataset
STORE_N = 10

# file settings:
DATASET_PATH = "/dataset"
DOWNLOAD_PATH = "/download"
VIDEO_PATH = "video/"

# supported video files:
SUPPORTED_FILES = ["mp4", "avi", "wmv", "mkv", "mov", "flv", "mpg", "m4v"]
IMG_TYPES = {"jpg", "png", "jpeg", "bmp", "tiff", "jp2", "sr", "ras"}

# control frame settings
FRAME_SETTINGS = {"bd": 0, "bg": GUI_BG}
BUTTON_SETTINGS = {"bg": GUI_BG, "activebackground": GUI_BG, "cursor": "hand2", "border": 0}
button_config = {'row': 0, 'padx': 2, 'pady': 5}

# determines size of preview images in dateset explorer
data_set_previewsize = 125

