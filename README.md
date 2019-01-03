# OpenCV-Video-Label
This Git repository implements a tkinter/opencv video player which allows users to play videos and enables them to test their own algorithms in a user friendly environment. Besides, it already supports two object tracking algorithms (Re3 and CMT) which allow users to label an object once, track the object over multiple frames and store the resulting cropped images of the object, this reduces the amount of time needed for image tagging which is usually needed when creating datasets.

# About the implemented Algorithms:
## [Re3](https://gitlab.com/danielgordon10/re3-tensorflow):
Re3 is a real-time recurrent regression tracker. It offers accuracy and robustness similar to other state-of-the-art trackers while operating at 150 FPS. For more details, contact xkcd@cs.washington.edu. 

This repository implements the training and testing procedure from https://arxiv.org/pdf/1705.06368.pdf. 
A sample of the tracker can be found here: https://youtu.be/RByCiOLlxug.

Re3 is released under the GPL V3.
Please cite Re3 in your publications if it helps your research:

    @article{gordon2018re3,
        title={Re3: Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects},
        author={Gordon, Daniel and Farhadi, Ali and Fox, Dieter},
        journal={IEEE Robotics and Automation Letters},
        volume={3},
        number={2},
        pages={788--795},
        year={2018},
        publisher={IEEE}
     }
 
## [CMT ](https://github.com/gnebehay/CMT):
CMT (Consensus-based Matching and Tracking of Keypoints for Object Tracking) is a novel keypoint-based method for long-term model-free object tracking in a combined matching-and-tracking framework. Details can be found on the project page and in their publication. 

If you use their algorithm in scientific work, please cite their publication:

    @inproceedings{Nebehay2015CVPR,
        author = {Nebehay, Georg and Pflugfelder, Roman},
        booktitle = {Computer Vision and Pattern Recognition},
        month = jun,
        publisher = {IEEE},
        title = {Clustering of {Static-Adaptive} Correspondences for Deformable Object Tracking},
        year = {2015}
    }
# Input types:
The tool currently supports the following video sources:
1. Regular video files like .mp4 and .avi
2. Webcams
3. IPwebcams (which allows users to stream from their phone)
4. Screencapturing (currently only to generate videos, which can then be used for tracking)

# Dependencies:
1. Python 3.5 
2. OpenCV 2 
2. Tensorflow and its requirements.
3. NumPy
4. SciPy
5. mss
6. Pillow
7. 
8. CUDA (Strongly recommended for Re3)
9. cuDNN (recommended for Re3)

# Installation:
 If you which to use the Re3 tracker please download their latest model (http://bit.ly/2L5deYF) and unzip its content in algorithms/re3/logs/ 
