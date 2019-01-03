import cv2
import numpy as np
import os
import tensorflow as tf
import sys
import os.path
from .tracker import network
from .re3_utils.util import bb_util
from .re3_utils.util import im_util
from .re3_utils.tensorflow_util import tf_util
# Network Constants
from .model_downloader import download_re3_model
from .constants import CROP_SIZE
from .constants import CROP_PAD
from .constants import LSTM_SIZE
from .constants import LOG_DIR
from .constants import GPU_ID
from .constants import MAX_TRACK_LENGTH

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

SPEED_OUTPUT = True


class Re3Tracker(object):
    def __init__(self, gpu_id=GPU_ID):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        basedir = os.path.dirname(__file__)
        tf.Graph().as_default()

        self.imagePlaceholder = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
        self.prevLstmState = tuple([tf.placeholder(tf.float32, shape=(None, LSTM_SIZE)) for _ in range(4)])
        self.batch_size = tf.placeholder(tf.int32, shape=())
        self.outputs, self.state1, self.state2 = network.inference(
            self.imagePlaceholder, num_unrolls=1, batch_size=self.batch_size, train=False,
            prevLstmState=self.prevLstmState)
        self.sess = tf_util.Session()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.join(basedir, '..', LOG_DIR, 'checkpoints'))
        if ckpt is None:
            download_re3_model()
            ckpt = tf.train.get_checkpoint_state(os.path.join(basedir, '..', LOG_DIR, 'checkpoints'))
            if ckpt is None:
                print("Something went wrong while downloading the model, please"
                      " try again or download the file manually using the following url:\n http://bit.ly/2L5deYF")
        tf_util.restore(self.sess, ckpt.model_checkpoint_path)
        self.tracked_data = {}
        self.total_forward_count = -1

    # unique_id{str}: A unique id for the object being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_box{None or 4x1 numpy array or list}: 4x1 bounding box in X1, Y1, X2, Y2 format.
    def track(self, unique_id, image, starting_box=None):

        if type(image) == str:
            image = cv2.imread(image)[:, :, ::-1]
        else:
            image = image.copy()

        if starting_box is not None:
            lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
            pastBBox = np.array(starting_box)  # turns list into numpy array if not and copies for safety.
            prevImage = image
            originalFeatures = None
            forwardCount = 0
        elif unique_id in self.tracked_data:
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
        else:
            raise Exception('Unique_id %s with no initial bounding box' % unique_id)

        croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
        croppedInput1, _ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)

        feed_dict = {
            self.imagePlaceholder: [croppedInput0, croppedInput1],
            self.prevLstmState: lstmState,
            self.batch_size: 1,
        }
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        lstmState = [s1[0], s1[1], s2[0], s2[1]]
        if forwardCount == 0:
            originalFeatures = [s1[0], s1[1], s2[0], s2[1]]

        # Shift output box to full image coordinate system.
        outputBox = bb_util.from_crop_coordinate_system(rawOutput.squeeze() / 10.0, pastBBoxPadded, 1, 1)

        if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
            croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
            input = np.tile(croppedInput[np.newaxis, ...], (2, 1, 1, 1))
            feed_dict = {
                self.imagePlaceholder: input,
                self.prevLstmState: originalFeatures,
                self.batch_size: 1,
            }
            rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
            lstmState = [s1[0], s1[1], s2[0], s2[1]]

        forwardCount += 1
        self.total_forward_count += 1

        if starting_box is not None:
            # Use label if it's given
            outputBox = np.array(starting_box)

        self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)

        return [max(int(i), 0) for i in outputBox]
