# Network Constants
CROP_SIZE = 227
CROP_PAD = 2
MAX_TRACK_LENGTH = 32
LSTM_SIZE = 512

import os.path
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
GPU_ID = '0'

# Drawing constants
resize_factor = 0.7
VIDEO_W = int(1250 * resize_factor)
VIDEO_H = int(720 * resize_factor)

OUTPUT_WIDTH = VIDEO_W
OUTPUT_HEIGHT = VIDEO_H
PADDING = 2
