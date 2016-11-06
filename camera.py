import numpy as np
import tensorflow as tf
import deep_dream
import cv2

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self, sess, t_input, t):
        success, frame = self.video.read()
        frame = np.float32(frame)

        output_frame = deep_dream.render_deepdream(sess, tf.square(t), t_input, img0=frame)

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', output_frame)
        return jpeg.tobytes()
